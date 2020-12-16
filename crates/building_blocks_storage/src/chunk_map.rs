//! A sparse lattice map made of up array chunks.
//!
//! # Addressing
//!
//! The data can either be addressed by chunk key with the `get_chunk*` methods or by individual points using the `Get*` and
//! `ForEach*` trait impls. The map of chunks uses `Point3i` keys. The key for a chunk is the minimum point in that chunk, which
//! is always a multiple of the chunk shape. Chunk shape dimensions must be powers of 2, which allows for efficiently
//! calculating a chunk key from any point in the chunk.
//!
//! # Chunk Storage
//!
//! `ChunkMap<N, T, Meta, Store>` depends on a backing chunk storage `Store`, which can implement some of `ChunkReadStorage` or
//! `ChunkWriteStorage`. A storage can be as simple as a `HashMap`, which provides good performance for both iteration and
//! random access. It could also be something more memory efficient like `CompressibleChunkStorage` or
//! `CompressibleChunkStorageReader`, which perform nearly as well but involve some extra management of the cache.
//!
//! # Serialization
//!
//! In order to efficiently serialize a `ChunkMap`, you can first use `SerializableChunkMap::from_chunk_map` to create a compact
//! serializable representation. It will compress the bincode representation of the chunks.
//!
//! # Example `ChunkHashMap` Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder {
//!    chunk_shape: PointN([16; 3]), // components must be powers of 2
//!    ambient_value,
//!    default_chunk_metadata: (), // chunk metadata is optional
//! };
//! let mut map = builder.build_with_hash_map_storage();
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted.
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
//!
//! # Example `CompressibleChunkMap` Usage
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! #
//! # let builder = ChunkMapBuilder {
//! #    chunk_shape: PointN([16; 3]), // components must be powers of 2
//! #    ambient_value: 0,
//! #    default_chunk_metadata: (), // chunk metadata is optional
//! # };
//! #
//! let mut map = builder.build(CompressibleChunkStorage::new(Lz4 { level: 10 }));
//!
//! // You can write voxels the same as any other `ChunkMap`. As chunks are created, they will be placed in an LRU cache.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Save some space by compressing the least recently used chunks. On further access to the compressed chunks, they will get
//! // decompressed and cached.
//! map.storage_mut().compress_lru();
//!
//! // In order to use the read-only access traits, you need to construct a `CompressibleChunkStorageReader`.
//! let local_cache = LocalChunkCache::new();
//! let reader = map.storage().reader(&local_cache);
//! let reader_map = builder.build(reader);
//!
//! let bounding_extent = reader_map.bounding_extent();
//! reader_map.for_each(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, 1);
//!     } else {
//!         assert_eq!(value, 0);
//!     }
//! });
//!
//! // For efficient caching, you should flush your local cache back into the main storage when you are done with it.
//! map.storage_mut().flush_local_cache(local_cache);
//! ```

mod ambient;
mod chunk;
mod storage;

pub use ambient::*;
pub use chunk::*;
pub use storage::*;

pub mod conditional_aliases {
    pub use super::storage::conditional_aliases::*;
}

use crate::{
    access::{
        ForEach, ForEachMut, GetUncheckedMutRelease, GetUncheckedRelease, ReadExtent, WriteExtent,
    },
    array::{ArrayCopySrc, ArrayIndexer, ArrayN},
    Get, GetMut,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerPoint, PointN};

use either::Either;
use fnv::FnvHashMap;
use serde::{Deserialize, Serialize};

/// A lattice map made up of same-shaped `ArrayN` chunks. It takes a value at every possible `PointN`, because accesses made
/// outside of the stored chunks will return some ambient value specified on creation.
///
/// `ChunkMap` is generic over the type used to actually store the `Chunk`s. You can use any storage that implements
/// `ChunkReadStorage` or `ChunkWriteStorage`.
pub struct ChunkMap<N, T, Meta, Store> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,

    ambient_value: T,
    default_chunk_metadata: Meta,
    storage: Store,
}

/// A 2-dimensional `ChunkMap`.
pub type ChunkMap2<T, Meta, Store> = ChunkMap<[i32; 2], T, Meta, Store>;
/// A 3-dimensional `ChunkMap`.
pub type ChunkMap3<T, Meta, Store> = ChunkMap<[i32; 3], T, Meta, Store>;

/// A few pieces of info used within the `ChunkMap`. You will probably keep one of these around to create new `ChunkMap`s from
/// a chunk storage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ChunkMapBuilder<N, T, Meta = ()> {
    /// The shape of each chunk.
    pub chunk_shape: PointN<N>,
    /// The value to use when none is specified, i.e. when creating new chunks or accessing vacant chunks.
    pub ambient_value: T,
    /// The metadata value used to initialize new chunks.
    pub default_chunk_metadata: Meta,
}

/// A 2-dimensional `ChunkMapBuilder`.
pub type ChunkMapBuilder2<T, Meta = ()> = ChunkMapBuilder<[i32; 2], T, Meta>;
/// A 3-dimensional `ChunkMapBuilder`.
pub type ChunkMapBuilder3<T, Meta = ()> = ChunkMapBuilder<[i32; 3], T, Meta>;

impl<N, T, Meta> ChunkMapBuilder<N, T, Meta>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
{
    /// Create a new `ChunkMap` with the given `storage`.
    pub fn build<Store>(self, storage: Store) -> ChunkMap<N, T, Meta, Store> {
        ChunkMap::new(
            self.chunk_shape,
            self.ambient_value,
            self.default_chunk_metadata,
            storage,
        )
    }

    /// Create a new `ChunkMap` using a `FnvHashMap` as the chunk storage.
    pub fn build_with_hash_map_storage(self) -> ChunkHashMap<N, T, Meta> {
        self.build(FnvHashMap::default())
    }
}

impl<'a, N, T, Meta, Store> ChunkMap<N, T, Meta, Store> {
    /// Consumes `self` and returns the backing chunk storage.
    #[inline]
    pub fn take_storage(self) -> Store {
        self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage(&self) -> &Store {
        &self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut Store {
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
    pub fn default_chunk_metadata(&self) -> &Meta {
        &self.default_chunk_metadata
    }
}

impl<'a, N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
{
    /// Creates a map using the given `storage`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    pub fn new(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: Meta,
        storage: Store,
    ) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            ambient_value,
            default_chunk_metadata,
            storage,
        }
    }

    /// Get the `ChunkMapBuilder` used to build this map.
    pub fn builder(&self) -> ChunkMapBuilder<N, T, Meta>
    where
        T: Copy,
        Meta: Clone,
    {
        ChunkMapBuilder {
            chunk_shape: self.indexer.chunk_shape(),
            ambient_value: self.ambient_value,
            default_chunk_metadata: self.default_chunk_metadata.clone(),
        }
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    Store: ChunkReadStorage<N, T, Meta>,
{
    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_chunk(&self, key: &PointN<N>) -> Option<&Chunk<N, T, Meta>> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get(key)
    }

    /// Returns the chunk containing `point` if it exists.
    #[inline]
    pub fn get_chunk_containing_point(
        &self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &Chunk<N, T, Meta>)> {
        let chunk_key = self.indexer.chunk_key_containing_point(point);

        self.get_chunk(&chunk_key).map(|c| (chunk_key, c))
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn write_chunk(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>) {
        debug_assert!(chunk.array.extent().shape.eq(&self.indexer.chunk_shape()));
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.write(key, chunk);
    }

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn replace_chunk(
        &mut self,
        key: PointN<N>,
        chunk: Chunk<N, T, Meta>,
    ) -> Option<Chunk<N, T, Meta>> {
        debug_assert!(chunk.array.extent().shape.eq(&self.indexer.chunk_shape()));
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.replace(key, chunk)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    fn get_mut_chunk(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, Meta>> {
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
        create_chunk: impl FnOnce() -> Chunk<N, T, Meta>,
    ) -> &mut Chunk<N, T, Meta> {
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.get_mut_or_insert_with(key, create_chunk)
    }

    /// Returns the mutable chunk containing `point` if it exists.
    #[inline]
    pub fn get_mut_chunk_containing_point(
        &mut self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &mut Chunk<N, T, Meta>)> {
        let chunk_key = self.indexer.chunk_key_containing_point(point);

        self.get_mut_chunk(&chunk_key).map(|c| (chunk_key, c))
    }

    /// Get mutable data for point `p` along with the chunk key. If `p` does not exist, calls `create_chunk` to fill that entry
    /// first.
    #[inline]
    pub fn get_mut_point_or_insert_chunk_with(
        &mut self,
        p: &PointN<N>,
        create_chunk: impl FnOnce(PointN<N>, ExtentN<N>) -> Chunk<N, T, Meta>,
    ) -> (PointN<N>, &mut T)
    where
        N: ArrayIndexer<N>,
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
        N: ArrayIndexer<N>,
        T: Copy,
        Meta: Clone,
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

impl<'a, N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    Store: IterChunkKeys<'a, N>,
{
    /// The smallest extent that bounds all chunks.
    pub fn bounding_extent(&'a self) -> ExtentN<N> {
        bounding_extent(self.storage.chunk_keys().flat_map(|key| {
            let chunk_extent = self.indexer.extent_for_chunk_at_key(*key);

            vec![chunk_extent.minimum, chunk_extent.max()].into_iter()
        }))
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, Meta, Store> Get<&PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    N: ArrayIndexer<N>,
    T: Copy,
    Store: ChunkReadStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn get(&self, p: &PointN<N>) -> Self::Data {
        self.get_chunk_containing_point(p)
            .map(|(_key, chunk)| chunk.array.get_unchecked_release(p))
            .unwrap_or(self.ambient_value)
    }
}

impl<'a, N, T, Meta, Store> GetMut<&PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    N: ArrayIndexer<N>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, p: &PointN<N>) -> &mut Self::Data {
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

impl<N, T, Meta, Store> ForEach<N, PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    ArrayN<N, T>: ForEach<N, PointN<N>, Data = T>,
    T: Copy,
    Store: ChunkReadStorage<N, T, Meta>,
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

impl<'a, N, T, Meta, Store> ForEachMut<N, PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    ArrayN<N, T>: ForEachMut<N, PointN<N>, Data = T>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
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

impl<'a, N, T, Meta, Store> ReadExtent<'a, N> for ChunkMap<N, T, Meta, Store>
where
    N: ArrayIndexer<N>,
    PointN<N>: 'a + IntegerPoint<N> + ChunkShape<N>,
    T: 'a + Copy,
    Store: ChunkReadStorage<N, T, Meta>,
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
impl<'a, N, T, Meta, Store, Src> WriteExtent<N, Src> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N> + ChunkShape<N>,
    ArrayN<N, T>: WriteExtent<N, Src>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
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

pub type ChunkCopySrc<Meta, N, T> = Either<ArrayCopySrc<Meta>, AmbientExtent<N, T>>;

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

    const BUILDER: ChunkMapBuilder<[i32; 3], i32, ()> = ChunkMapBuilder {
        chunk_shape: PointN([16; 3]),
        ambient_value: 0,
        default_chunk_metadata: (),
    };

    #[test]
    fn write_and_read_points() {
        let mut map = BUILDER.build_with_hash_map_storage();

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
        let mut map = BUILDER.build_with_hash_map_storage();

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

        let mut map = BUILDER.build_with_hash_map_storage();

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
