//! A sparse lattice map made of up array chunks.
//!
//! # Indexing and Iteration
//!
//! The data can either be addressed by chunk key with the `get_chunk*` methods or by individual points using the `Get*` and
//! `ForEach*` trait impls. The map of chunks uses `Point3i` keys. The key for a chunk is the minimum point in that chunk, which
//! is always a multiple of the chunk shape. Chunk shape dimensions must be powers of 2, which allows for efficiently
//! calculating a chunk key from any point in the chunk.
//!
//! If you require iteration over large, but very sparse regions, you might want an additional `OctreeChunkIndex` to track the
//! set of occupied chunks. Traversing that index can be faster than doing hash map lookups on all of the possible chunks in a
//! region.
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
//! In order to efficiently serialize a `ChunkMap`, you can first use `SerializableChunks::from_chunk_map` to create a compact
//! serializable representation. It will compress the bincode representation of the chunks.
//!
//! # Example `ChunkHashMap` Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let ambient_value = 0;
//! let chunk_shape = Point3i::fill(16);
//! let mut map = ChunkMap3x1::build_with_hash_map_storage(chunk_shape);
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *map.get_mut(p) = 1;
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
//! // You can also access individual points like you can with a `ArrayNx1`. This is
//! // slower than iterating, because it hashes the chunk coordinates for every access.
//! for &p in write_points.iter() {
//!     assert_eq!(map.get(p), 1);
//! }
//! assert_eq!(map.get(Point3i::fill(1)), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(32));
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
//! # let chunk_shape = Point3i::fill(16);
//! #
//! let mut map = ChunkMap::build_with_write_storage(
//!     chunk_shape,
//!     FastCompressibleChunkStorage::with_bytes_compression(Lz4 { level: 10 })
//! );
//!
//! // You can write voxels the same as any other `ChunkMap`. As chunks are created, they will be placed in an LRU cache.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *map.get_mut(p) = 1;
//! }
//!
//! // Save some space by compressing the least recently used chunks. On further access to the compressed chunks, they will get
//! // decompressed and cached.
//! map.storage_mut().compress_lru();
//!
//! // In order to use the read-only access traits, you need to construct a `CompressibleChunkStorageReader`.
//! let local_cache = LocalChunkCache::new();
//! let reader = map.reader(&local_cache);
//!
//! let bounding_extent = reader.bounding_extent();
//! reader.for_each(&bounding_extent, |p, value| {
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

use crate::{
    ArrayCopySrc, ArrayIndexer, ArrayNx1, Chunk, ChunkHashMap, ChunkIndexer, ChunkReadStorage,
    ChunkWriteStorage, ForEach, ForEachMut, Get, GetMut, GetRef, GetUnchecked,
    GetUncheckedMutRelease, GetUncheckedRef, GetUncheckedRefRelease, IterChunkKeys, ReadExtent,
    SmallKeyHashMap, WriteExtent,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerPoint, PointN};

use core::hash::Hash;
use either::Either;

/// A lattice map made up of same-shaped `ArrayNx1` chunks. It takes a value at every possible `PointN`, because accesses made
/// outside of the stored chunks will return some ambient value specified on creation.
///
/// `ChunkMap` is generic over the type used to actually store the `Chunk`s. You can use any storage that implements
/// `ChunkReadStorage` or `ChunkWriteStorage`. Being a lattice map, `ChunkMap` will implement various access traits, depending
/// on the capabilities of the chunk storage.
///
/// If the chunk storage implements `ChunkReadStorage`, then `ChunkMap` will implement:
/// - `Get`
/// - `ForEach`
/// - `ReadExtent`
///
/// If the chunk storage implements `ChunkWriteStorage`, then `ChunkMap` will implement:
/// - `GetMut`
/// - `ForEachMut`
/// - `WriteExtent`
pub struct ChunkMap<N, T, Ch, Store> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,
    storage: Store,
    ambient_value: T, // Needed for GetRef to return a reference to non-temporary value

    marker: std::marker::PhantomData<Ch>,
}

/// A 2-dimensional `ChunkMap`.
pub type ChunkMap2<T, Ch, Store> = ChunkMap<[i32; 2], T, Ch, Store>;
/// A 3-dimensional `ChunkMap`.
pub type ChunkMap3<T, Ch, Store> = ChunkMap<[i32; 3], T, Ch, Store>;

/// An N-dimensional, single-channel `ChunkMap`.
pub type ChunkMapNx1<N, T, Store> = ChunkMap<N, T, ArrayNx1<N, T>, Store>;
/// A 2-dimensional, single-channel `ChunkMap`.
pub type ChunkMap2x1<T, Store> = ChunkMapNx1<[i32; 2], T, Store>;
/// A 3-dimensional, single-channel `ChunkMap`.
pub type ChunkMap3x1<T, Store> = ChunkMapNx1<[i32; 3], T, Store>;

impl<N, T, Ch, Store> ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
{
    /// Create a new `ChunkMap` with the given `storage` which must implement both `ChunkReadStorage` and `ChunkWriteStorage`.
    pub fn build_with_rw_storage(chunk_shape: PointN<N>, storage: Store) -> Self
    where
        Store: ChunkReadStorage<N, Ch> + ChunkWriteStorage<N, Ch>,
    {
        Self::new(chunk_shape, storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkReadStorage`.
    pub fn build_with_read_storage(chunk_shape: PointN<N>, storage: Store) -> Self
    where
        Store: ChunkReadStorage<N, Ch>,
    {
        Self::new(chunk_shape, storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkWriteStorage`.
    pub fn build_with_write_storage(chunk_shape: PointN<N>, storage: Store) -> Self
    where
        Store: ChunkWriteStorage<N, Ch>,
    {
        Self::new(chunk_shape, storage)
    }
}

impl<N, T, Ch> ChunkHashMap<N, T, Ch>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Ch: Chunk<N, T>,
{
    /// Create a new `ChunkMap` using a `SmallKeyHashMap` as the chunk storage.
    pub fn build_with_hash_map_storage(chunk_shape: PointN<N>) -> Self {
        Self::build_with_rw_storage(chunk_shape, SmallKeyHashMap::default())
    }
}

impl<N, T, Ch, Store> ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
{
    /// Creates a map using the given `storage`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    fn new(chunk_shape: PointN<N>, storage: Store) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            storage,
            ambient_value: Ch::ambient_value(),
            marker: Default::default(),
        }
    }
}

impl<N, T, Ch, Store> ChunkMap<N, T, Ch, Store> {
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
}

impl<N, T, Ch, Store> ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Store: ChunkReadStorage<N, Ch>,
{
    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_chunk(&self, key: PointN<N>) -> Option<&Ch> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get(key)
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be represented by an `AmbientExtent`.
    #[inline]
    pub fn visit_chunks(
        &self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(Either<&Ch, (&ExtentN<N>, AmbientExtent<N, T>)>),
    ) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(chunk_key) {
                visitor(Either::Left(chunk))
            } else {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(chunk_key);
                visitor(Either::Right((
                    &chunk_extent,
                    AmbientExtent::new(Ch::ambient_value()),
                )))
            }
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_chunks(&self, extent: &ExtentN<N>, mut visitor: impl FnMut(&Ch)) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(chunk_key) {
                visitor(chunk)
            }
        }
    }
}

impl<N, T, Ch, Store> ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Store: ChunkWriteStorage<N, Ch>,
{
    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn write_chunk(&mut self, key: PointN<N>, chunk: Ch) {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.write(key, chunk);
    }

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn replace_chunk(&mut self, key: PointN<N>, chunk: Ch) -> Option<Ch> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.replace(key, chunk)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk(&mut self, key: PointN<N>) -> Option<&mut Ch> {
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
        create_chunk: impl FnOnce() -> Ch,
    ) -> &mut Ch {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get_mut_or_insert_with(key, create_chunk)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, a new chunk is created with the ambient value.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_ambient(&mut self, key: PointN<N>) -> &mut Ch {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        let Self {
            indexer, storage, ..
        } = self;

        storage.get_mut_or_insert_with(key, || {
            Ch::new_ambient(indexer.extent_for_chunk_at_key(key))
        })
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be created first with ambient value.
    #[inline]
    pub fn visit_mut_chunks(&mut self, extent: &ExtentN<N>, mut visitor: impl FnMut(&mut Ch)) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            visitor(self.get_mut_chunk_or_insert_ambient(chunk_key));
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_mut_chunks(
        &mut self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Ch),
    ) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_mut_chunk(chunk_key) {
                visitor(chunk)
            }
        }
    }

    /// Fill all of `extent` with the same `value`.
    #[inline]
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T)
    where
        for<'r> Self: ForEachMut<'r, N, PointN<N>, Item = &'r mut T>,
        T: Clone,
    {
        self.for_each_mut(extent, |_p, v| *v = value.clone());
    }
}

impl<'a, N, T, Ch, Store> ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
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

/// An extent that takes the same value everywhere.
#[derive(Copy, Clone)]
pub struct AmbientExtent<N, T> {
    pub value: T,
    _n: std::marker::PhantomData<N>,
}

impl<N, T> AmbientExtent<N, T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _n: Default::default(),
        }
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.value.clone()
    }
}

impl<'a, N, T> ForEach<N, PointN<N>> for AmbientExtent<N, T>
where
    T: Clone,
    PointN<N>: IntegerPoint<N>,
{
    type Item = T;

    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        for p in extent.iter_points() {
            f(p, self.value.clone());
        }
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, Ch, Store> GetRef<PointN<N>, T> for ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Ch::Array: GetUncheckedRefRelease<PointN<N>, T>,
    Store: ChunkReadStorage<N, Ch>,
{
    #[inline]
    fn get_ref(&self, p: PointN<N>) -> &T {
        let key = self.indexer.chunk_key_containing_point(p);

        self.get_chunk(key)
            .map(|chunk| chunk.array_ref().get_unchecked_ref_release(p))
            .unwrap_or(&self.ambient_value)
    }
}

impl<N, T, Ch, Store> GetMut<PointN<N>, T> for ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Ch::Array: GetUncheckedMutRelease<PointN<N>, T>,
    Store: ChunkWriteStorage<N, Ch>,
{
    #[inline]
    fn get_mut(&mut self, p: PointN<N>) -> &mut T {
        let key = self.indexer.chunk_key_containing_point(p);
        let chunk = self.get_mut_chunk_or_insert_ambient(key);

        chunk.array_mut().get_unchecked_mut_release(p)
    }
}

impl_get_via_get_ref_and_clone!(ChunkMap<N, T, Ch, Store>, N, T, Ch, Store);

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<N, T, Ch, Store> ForEach<N, PointN<N>> for ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Ch::Array: ForEach<N, PointN<N>, Item = T>,
    T: Copy,
    Store: ChunkReadStorage<N, Ch>,
{
    type Item = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.visit_chunks(extent, |chunk| match chunk {
            Either::Left(chunk) => {
                chunk.array_ref().for_each(extent, |p, value| f(p, value));
            }
            Either::Right((chunk_extent, ambient)) => {
                ambient.for_each(&extent.intersection(&chunk_extent), |p, value| f(p, value))
            }
        });
    }
}

impl<'a, N, T, Ch, Store> ForEachMut<'a, N, PointN<N>> for ChunkMap<N, T, Ch, Store>
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Ch::Array: ForEachMut<'a, N, PointN<N>, Item = &'a mut T>,
    T: 'a,
    Store: ChunkWriteStorage<N, Ch>,
{
    type Item = &'a mut T;

    #[inline]
    fn for_each_mut(&'a mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.visit_mut_chunks(extent, |chunk| {
            // Tell the borrow checker that we're only giving out non-overlapping references.
            (unsafe { &mut *(chunk as *mut Ch) })
                .array_mut()
                .for_each_mut(extent, |p, value| f(p, value))
        });
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, N, T, Ch, Store> ReadExtent<'a, N> for ChunkMap<N, T, Ch, Store>
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    Ch: 'a + Chunk<N, T>,
    T: 'a + Copy,
    Store: ChunkReadStorage<N, Ch>,
{
    type Src = ChunkCopySrc<N, T, &'a Ch>;
    type SrcIter = ChunkCopySrcIter<N, T, &'a Ch>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .indexer
            .chunk_keys_for_extent(extent)
            .map(|key| {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(key);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.get_chunk(key)
                        .map(|chunk| Either::Left(ArrayCopySrc(chunk)))
                        .unwrap_or_else(|| Either::Right(AmbientExtent::new(Ch::ambient_value()))),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If ArrayNx1 supports writing from type Src, then so does ChunkMap.
impl<N, T, Ch, Store, Src> WriteExtent<N, Src> for ChunkMap<N, T, Ch, Store>
where
    PointN<N>: IntegerPoint<N>,
    Ch: Chunk<N, T>,
    Ch::Array: WriteExtent<N, Src>,
    Store: ChunkWriteStorage<N, Ch>,
    Src: Copy,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        self.visit_mut_chunks(extent, |chunk| chunk.array_mut().write_extent(extent, src));
    }
}

#[doc(hidden)]
pub type ChunkCopySrc<N, T, Ch> = Either<ArrayCopySrc<Ch>, AmbientExtent<N, T>>;
#[doc(hidden)]
pub type ChunkCopySrcIter<N, T, Ch> = std::vec::IntoIter<(ExtentN<N>, ChunkCopySrc<N, T, Ch>)>;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{copy_extent, Array3, Get};

    use building_blocks_core::prelude::*;

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);

    #[test]
    fn write_and_read_points() {
        let mut map = ChunkMap3x1::build_with_hash_map_storage(CHUNK_SHAPE);

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
            assert_eq!(map.get_mut(PointN(p)), &mut 0);
            *map.get_mut(PointN(p)) = 1;
            assert_eq!(map.get_mut(PointN(p)), &mut 1);
        }
    }

    #[test]
    fn write_extent_with_for_each_then_read() {
        let mut map = ChunkMap3x1::build_with_hash_map_storage(CHUNK_SHAPE);

        let write_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        map.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if write_extent.contains(p) {
                assert_eq!(map.get(p), 1);
            } else {
                assert_eq!(map.get(p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        let array = Array3::fill(extent_to_copy, 1);

        let mut map = ChunkMap3x1::build_with_hash_map_storage(CHUNK_SHAPE);

        copy_extent(&extent_to_copy, &array, &mut map);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(p) {
                assert_eq!(map.get(p), 1);
            } else {
                assert_eq!(map.get(p), 0);
            }
        }
    }
}
