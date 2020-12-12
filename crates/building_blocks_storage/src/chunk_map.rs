//! A sparse lattice map made of up array chunks.
//!
//! The data can either be addressed by chunk key with the `get_chunk*` methods or by individual points using the `Get*` and
//! `ForEach*` trait impls. The map of chunks uses `Point3i` keys. The key for a chunk is the minimum point in that chunk, which
//! is always a multiple of the chunk shape. Chunk shape dimensions must be powers of 2, which allows for efficiently
//! calculating a chunk key from any point in the chunk.
//!
//! `ChunkMap<N, T, M, S>` depends on a backing `S: ChunkStorage`. This can be as simple as a `HashMap`, which provides good
//! performance for both iteration and random access. It could also be something more memory efficient like `LruCacheStorage`,
//! which also performs well for iteration, but the caching overhead has an impact on random access performance.
//!
//! # Example Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let chunk_shape = PointN([16; 3]); // components must be powers of 2
//! let ambient_value = 0;
//! let default_chunk_meta = (); // chunk metadata is optional
//! let mut map = ChunkMap::with_hash_map_storage(chunk_shape, ambient_value, default_chunk_meta);
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted and cached.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Even though the map is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks.
//! let bounding_extent = map.bounding_extent();
//!
//! // Now we can read back the values.
//! map.for_each(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, 1);
//!     } else {
//!         // The points that we didn't write explicitly got an ambient value when the chunk was
//!         // inserted. Also any points in `bounding_extent` that don't have a chunk will also take
//!         // the ambient value.
//!         assert_eq!(value, 0);
//!     }
//! });
//!
//! // You can also access individual points like you can with a `ArrayN`. This is
//! // slower than iterating, because it hashes the chunk coordinates for every access.
//! for p in write_points.iter() {
//!     assert_eq!(map.get(p), 1);
//! }
//! assert_eq!(map.get(&PointN([1, 1, 1])), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([32; 3]));
//! let mut dense_map = Array3::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &map, &mut dense_map);
//! ```

mod ambient;
mod chunk;
mod storage;

pub use ambient::*;
pub use chunk::*;
pub use storage::*;

use crate::{
    access::{
        ForEach, ForEachMut, GetUncheckedMutRelease, GetUncheckedRelease, ReadExtent, WriteExtent,
    },
    array::{Array, ArrayCopySrc, ArrayN},
    Get, GetMut,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerExtent, IntegerPoint, PointN};

use either::Either;

/// A lattice map made up of same-shaped `ArrayN` chunks. It takes a value at every possible `PointN`, because accesses made
/// outside of the stored chunks will return some ambient value specified on creation.
///
/// Implemented with a hash map from "chunk key" to chunk, where the key is defined by `ChunkIndexer`.
///
/// When used as a cache, it's possible for a chunk to be `CacheState::Evicted`. This implies that the chunk must be fetched
/// from somewhere else.
pub struct ChunkMap<N, T, M, S> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,

    // The value to use when none is specified, i.e. when filling new chunks or erasing points.
    ambient_value: T,

    default_chunk_metadata: M,

    storage: S,
}

impl<'a, N, T, M, S> ChunkMap<N, T, M, S> {
    /// Consumes `self` and returns the backing `ChunkStorage`.
    #[inline]
    pub fn take_storage(self) -> S {
        self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Return the ambient value of the map.
    #[inline]
    pub fn ambient_value(&self) -> T
    where
        T: Copy,
    {
        self.ambient_value
    }

    /// Return the default metadata for chunks.
    #[inline]
    pub fn default_chunk_metadata(&self) -> &M {
        &self.default_chunk_metadata
    }
}

impl<'a, N, T, M, S> ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    /// Creates a map using the given `storage`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    pub fn new(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: M,
        storage: S,
    ) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            ambient_value,
            default_chunk_metadata,
            storage,
        }
    }

    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    fn get_chunk(&self, key: &PointN<N>) -> Option<&Chunk<N, T, M>> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get(key)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    fn get_mut_chunk(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, M>> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get_mut(key)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, `create_chunk` is called to insert one.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M> {
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.get_mut_or_insert_with(key, create_chunk)
    }

    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn insert_chunk(
        &mut self,
        key: PointN<N>,
        chunk: Chunk<N, T, M>,
    ) -> Option<Chunk<N, T, M>> {
        debug_assert!(chunk.array.extent().shape.eq(&self.indexer.chunk_shape()));
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.insert(key, chunk)
    }

    /// Returns the chunk containing `point` if it exists.
    #[inline]
    pub fn get_chunk_containing_point(
        &self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &Chunk<N, T, M>)> {
        let chunk_key = self.indexer.chunk_key_containing_point(point);

        self.get_chunk(&chunk_key).map(|c| (chunk_key, c))
    }

    /// Returns the mutable chunk containing `point` if it exists.
    #[inline]
    pub fn get_mut_chunk_containing_point(
        &mut self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &mut Chunk<N, T, M>)> {
        let chunk_key = self.indexer.chunk_key_containing_point(point);

        self.get_mut_chunk(&chunk_key).map(|c| (chunk_key, c))
    }

    /// The smallest extent that bounds all chunks.
    pub fn bounding_extent(&'a self) -> ExtentN<N> {
        bounding_extent(self.storage.iter_keys().flat_map(|key| {
            let chunk_extent = self.indexer.extent_for_chunk_at_key(*key);

            vec![chunk_extent.minimum, chunk_extent.max()].into_iter()
        }))
    }

    /// Get mutable data for point `p` along with the chunk key. If `p` does not exist, calls `create_chunk` to fill that entry
    /// first.
    #[inline]
    pub fn get_mut_point_or_insert_chunk_with(
        &mut self,
        p: &PointN<N>,
        create_chunk: impl FnOnce(PointN<N>, ExtentN<N>) -> Chunk<N, T, M>,
    ) -> (PointN<N>, &mut T)
    where
        ArrayN<N, T>: Array<N>,
    {
        let key = self.indexer.chunk_key_containing_point(p);
        let Self {
            indexer, storage, ..
        } = self;
        let chunk = storage.get_mut_or_insert_with(key, || {
            create_chunk(key, indexer.extent_for_chunk_at_key(key))
        });

        (key, chunk.array.get_unchecked_mut_release(p))
    }

    /// Sets point `p` to value `T`. If `p` is in a chunk that doesn't exist yet, then the chunk will first be filled with the
    /// ambient value and default metadata.
    #[inline]
    pub fn get_mut_point_and_chunk_key(&mut self, p: &PointN<N>) -> (PointN<N>, &mut T)
    where
        ArrayN<N, T>: Array<N>,
        T: Copy,
        M: Clone,
    {
        let key = self.indexer.chunk_key_containing_point(p);
        let Self {
            indexer,
            ambient_value,
            default_chunk_metadata,
            storage,
            ..
        } = self;
        let array = &mut storage
            .get_mut_or_insert_with(key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(indexer.extent_for_chunk_at_key(key), *ambient_value),
            })
            .array;

        (key, array.get_unchecked_mut_release(p))
    }

    /// Fill all of `extent` with the same `value`.
    #[inline]
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T)
    where
        Self: ForEachMut<N, PointN<N>, Data = T>,
        T: Copy,
    {
        self.for_each_mut(extent, |_p, v| *v = value);
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, N, T, M, S> Get<&PointN<N>> for ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N>,
    T: Copy,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    type Data = T;

    #[inline]
    fn get(&self, p: &PointN<N>) -> Self::Data {
        self.get_chunk_containing_point(p)
            .map(|(_key, chunk)| chunk.array.get_unchecked_release(p))
            .unwrap_or(self.ambient_value)
    }
}

impl<'a, N, T, M, S> GetMut<&PointN<N>> for ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N>,
    T: Copy,
    M: Clone,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, p: &PointN<N>) -> &mut T {
        let (_chunk_key, value_mut) = self.get_mut_point_and_chunk_key(p);

        value_mut
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<'a, N, T, M, S> ForEach<N, PointN<N>> for ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N> + ForEach<N, PointN<N>, Data = T>,
    T: Copy,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    type Data = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Data)) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(&chunk_key) {
                chunk.array.for_each(extent, |p, value| f(p, value));
            } else {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(chunk_key);
                AmbientExtent::new(self.ambient_value)
                    .for_each(&extent.intersection(&chunk_extent), |p, value| f(p, value));
            }
        }
    }
}

impl<'a, N, T, M, S> ForEachMut<N, PointN<N>> for ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: ForEachMut<N, PointN<N>, Data = T>,
    T: Copy,
    M: Clone,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    type Data = T;

    #[inline]
    fn for_each_mut(&mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &mut Self::Data)) {
        let Self {
            indexer,
            ambient_value,
            default_chunk_metadata,
            storage,
            ..
        } = self;

        for chunk_key in indexer.chunk_keys_for_extent(extent) {
            let chunk = storage.get_mut_or_insert_with(chunk_key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(indexer.extent_for_chunk_at_key(chunk_key), *ambient_value),
            });
            chunk.array.for_each_mut(extent, |p, value| f(p, value));
        }
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, N, T, M, S> ReadExtent<'a, N> for ChunkMap<N, T, M, S>
where
    ArrayN<N, T>: Array<N>,
    PointN<N>: 'a + IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    T: 'a + Copy,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
{
    type Src = ArrayChunkCopySrc<'a, N, T>;
    type SrcIter = ArrayChunkCopySrcIter<'a, N, T>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .indexer
            .chunk_keys_for_extent(extent)
            .map(|key| {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(key);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.get_chunk(&key)
                        .map(|chunk| Either::Left(ArrayCopySrc(&chunk.array)))
                        .unwrap_or_else(|| Either::Right(AmbientExtent::new(self.ambient_value))),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If ArrayN supports writing from type Src, then so does ChunkMap.
impl<'a, N, T, M, S, Src> WriteExtent<N, Src> for ChunkMap<N, T, M, S>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: WriteExtent<N, Src>,
    Chunk<N, T, M>: 'a,
    T: Copy,
    M: Clone,
    Chunk<N, T, M>: 'a,
    S: ChunkStorage<'a, N, T, M>,
    Src: Copy,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        let Self {
            indexer,
            ambient_value,
            default_chunk_metadata,
            storage,
            ..
        } = self;

        for chunk_key in indexer.chunk_keys_for_extent(extent) {
            let chunk = storage.get_mut_or_insert_with(chunk_key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(indexer.extent_for_chunk_at_key(chunk_key), *ambient_value),
            });
            chunk.array.write_extent(extent, src);
        }
    }
}

pub type ChunkCopySrc<M, N, T> = Either<ArrayCopySrc<M>, AmbientExtent<N, T>>;

pub type ArrayChunkCopySrcIter<'a, N, T> =
    std::vec::IntoIter<(ExtentN<N>, ArrayChunkCopySrc<'a, N, T>)>;
pub type ArrayChunkCopySrc<'a, N, T> = Either<ArrayCopySrc<&'a ArrayN<N, T>>, AmbientExtent<N, T>>;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{access::Get, copy_extent, Array3};

    use building_blocks_core::Extent3i;

    #[test]
    fn write_and_read_points() {
        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::with_hash_map_storage(chunk_shape, ambient_value, ());

        let points = [
            [0, 0, 0],
            [1, 2, 3],
            [16, 0, 0],
            [0, 16, 0],
            [0, 0, 16],
            [15, 0, 0],
            [-15, 0, 0],
        ];

        for p in points.iter().cloned() {
            assert_eq!(map.get_mut(&PointN(p)), &mut 0);
            *map.get_mut(&PointN(p)) = 1;
            assert_eq!(map.get_mut(&PointN(p)), &mut 1);
        }
    }

    #[test]
    fn write_extent_with_for_each_then_read() {
        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::with_hash_map_storage(chunk_shape, ambient_value, ());

        let write_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        map.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        for p in read_extent.iter_points() {
            if write_extent.contains(&p) {
                assert_eq!(map.get(&p), 1);
            } else {
                assert_eq!(map.get(&p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        let array = Array3::fill(extent_to_copy, 1);

        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::with_hash_map_storage(chunk_shape, ambient_value, ());

        copy_extent(&extent_to_copy, &array, &mut map);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(&p) {
                assert_eq!(map.get(&p), 1);
            } else {
                assert_eq!(map.get(&p), 0);
            }
        }
    }
}
