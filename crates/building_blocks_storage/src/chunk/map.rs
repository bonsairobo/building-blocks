//! A sparse lattice map made of up array chunks. Designed for random access. Supports multiple levels of detail.
//!
//! # Level of Detail
//!
//! All chunks have the same shape, but the voxel size doubles at every level. Most of the time, you will just manipulate the
//! data at LOD0, but if you need to downsample this to save resources where coarser resolution is acceptable, then you can use
//! a `ChunkDownsampler` and the `ChunkMap::downsample_*` methods to populate higher levels.
//!
//! *NOTE*: If you want your downsampled data to have different number of channels than LOD0, then you will need to store the
//! downsampled chunks in a different `ChunkMap`. You will need to use specialized methods for this use case:
//! - `ChunkMap::downsample_external_chunk`
//! - `ChunkMap::downsample_chunks_with_lod0_and_index`
//!
//! # Indexing and Iteration
//!
//! The data can either be addressed by `ChunkKey` with the `get_chunk*` methods or by individual points using the `Get*` and
//! `ForEach*` trait impls on a `ChunkMapLodView`. At a given level of detail, the key for a chunk is the minimum point in that
//! chunk, which is always a multiple of the chunk shape. Chunk shape dimensions must be powers of 2, which allows for
//! efficiently calculating a chunk minimum from any point in the chunk.
//!
//! If you require iteration over large, but very sparse regions, you might want an additional `OctreeChunkIndex` to track the
//! set of occupied chunks. Traversing that index can be faster than doing hash map lookups on all of the possible chunks in a
//! region.
//!
//! When using multiple levels of detail, it's generally useful to have an `OctreeChunkIndex` that tracks the chunk occupancy.
//! This helps with specific types of traversal, like for downsampling or detecting clipmap updates.
//!
//! # Chunk Storage
//!
//! `ChunkMap<N, T, Bldr, Store>` depends on a backing chunk storage `Store`, which can implement some of `ChunkReadStorage` or
//! `ChunkWriteStorage`. A storage can be as simple as a `HashMap`, which provides good performance for both iteration and
//! random access. It could also be something more memory efficient like `FastCompressibleChunkStorage` or
//! `CompressibleChunkStorageReader`, which perform nearly as well but involve some extra management of the cache.
//!
//! # Serialization
//!
//! In order to efficiently serialize a `ChunkMap`, you can first use `SerializableChunks::from_iter` to create a compact
//! serializable representation. It will compress the bincode representation of the chunks.
//!
//! # Example `ChunkHashMap` Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let chunk_shape = Point3i::fill(16);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(chunk_shape, ambient_value);
//! let mut map = builder.build_with_hash_map_storage();
//!
//! // We need to focus on a specific level of detail to use the access traits.
//! let mut lod0 = map.lod_view_mut(0);
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *lod0.get_mut(p) = 1;
//! }
//!
//! // Even though the map is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks in LOD0.
//! let bounding_extent = map.bounding_extent(0);
//!
//! // Now we can read back the values.
//! let lod0 = map.lod_view(0);
//! lod0.for_each(&bounding_extent, |p, value| {
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
//! // You can also access individual points like you can with an `Array`. This is
//! // slower than iterating, because it hashes the chunk coordinates for every access.
//! for &p in write_points.iter() {
//!     assert_eq!(lod0.get(p), 1);
//! }
//! assert_eq!(lod0.get(Point3i::fill(1)), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(32));
//! let mut dense_map = Array3x1::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &lod0, &mut dense_map);
//! ```
//!
//! # Example `CompressibleChunkMap` Usage
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! #
//! let chunk_shape = Point3i::fill(16);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(chunk_shape, ambient_value);
//! let mut map = builder.build_with_write_storage(
//!     FastCompressibleChunkStorageNx1::with_bytes_compression(Lz4 { level: 10 })
//! );
//! let mut lod0 = map.lod_view_mut(0);
//!
//! // You can write voxels the same as any other `ChunkMap`. As chunks are created, they will be placed in an LRU cache.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *lod0.get_mut(p) = 1;
//! }
//!
//! // Save some space by compressing the least recently used chunks. On further access to the compressed chunks, they will get
//! // decompressed and cached.
//! map.storage_mut().compress_lru();
//!
//! // In order to use the read-only access traits, you need to construct a `CompressibleChunkStorageReader`.
//! let local_cache = LocalChunkCache3::new();
//! let reader = map.reader(&local_cache);
//!
//! let bounding_extent = reader.bounding_extent(0);
//! reader.lod_view(0).for_each(&bounding_extent, |p, value| {
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

pub mod builder;
pub mod lod_view;
pub mod sampling;

pub use builder::*;
pub use lod_view::*;
pub use sampling::*;

use crate::{
    Array, ChunkIndexer, ChunkKey, ChunkReadStorage, ChunkWriteStorage, FillExtent, ForEach, Get,
    GetMut, GetRef, IterChunkKeys, MultiRef,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerPoint, PointN};

use either::Either;

/// One piece of a chunked lattice map.
pub trait Chunk {
    /// The inner array type. This makes it easier for `Chunk` implementations to satisfy access trait bounds by inheriting them
    /// from existing array types like `Array`.
    type Array;

    /// Borrow the inner array.
    fn array(&self) -> &Self::Array;

    /// Mutably borrow the inner array.
    fn array_mut(&mut self) -> &mut Self::Array;
}

impl<N, Chan> Chunk for Array<N, Chan> {
    type Array = Self;

    #[inline]
    fn array(&self) -> &Self::Array {
        self
    }

    #[inline]
    fn array_mut(&mut self) -> &mut Self::Array {
        self
    }
}

/// A lattice map made up of same-shaped `Array` chunks. For each level of detail, it takes a value at every possible `PointN`,
/// because accesses made outside of the stored chunks will return some ambient value specified on creation.
///
/// `ChunkMap` is generic over the type used to actually store the `Chunk`s. You can use any storage that implements
/// `ChunkReadStorage` or `ChunkWriteStorage`. Being a lattice map, `ChunkMapLodView` will implement various access traits,
/// depending on the capabilities of the chunk storage.
///
/// If the chunk storage implements `ChunkReadStorage`, then `ChunkMapLodView` will implement:
/// - `Get`
/// - `ForEach`
/// - `ReadExtent`
///
/// If the chunk storage implements `ChunkWriteStorage`, then `ChunkMapLodView` will implement:
/// - `GetMut`
/// - `ForEachMut`
/// - `WriteExtent`
pub struct ChunkMap<N, T, Bldr, Store> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,
    storage: Store,
    builder: Bldr,
    ambient_value: T, // Needed for GetRef to return a reference to non-temporary value
}

/// A 2-dimensional `ChunkMap`.
pub type ChunkMap2<T, Bldr, Store> = ChunkMap<[i32; 2], T, Bldr, Store>;
/// A 3-dimensional `ChunkMap`.
pub type ChunkMap3<T, Bldr, Store> = ChunkMap<[i32; 3], T, Bldr, Store>;

/// An N-dimensional, single-channel `ChunkMap`.
pub type ChunkMapNx1<N, T, Store> = ChunkMap<N, T, ChunkMapBuilderNx1<N, T>, Store>;
/// A 2-dimensional, single-channel `ChunkMap`.
pub type ChunkMap2x1<T, Store> = ChunkMapNx1<[i32; 2], T, Store>;
/// A 3-dimensional, single-channel `ChunkMap`.
pub type ChunkMap3x1<T, Store> = ChunkMapNx1<[i32; 3], T, Store>;

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
{
    /// Creates a map using the given `storage`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    fn new(builder: Bldr, storage: Store) -> Self {
        let indexer = ChunkIndexer::new(builder.chunk_shape());
        let ambient_value = builder.ambient_value();

        Self {
            indexer,
            storage,
            builder,
            ambient_value,
        }
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store> {
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

    #[inline]
    pub fn builder(&self) -> &Bldr {
        &self.builder
    }

    /// Get an immutable view of a single level of detail `lod` in order to use the access traits.
    #[inline]
    pub fn lod_view(&self, lod: u8) -> ChunkMapLodView<&'_ Self> {
        ChunkMapLodView {
            delegate: self,
            lod,
        }
    }

    /// Get a mutable view of a single level of detail `lod` in order to use the access traits.
    #[inline]
    pub fn lod_view_mut(&mut self, lod: u8) -> ChunkMapLodView<&'_ mut Self> {
        ChunkMapLodView {
            delegate: self,
            lod,
        }
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    Bldr: ChunkMapBuilder<N, T>,
{
    #[inline]
    pub fn chunk_shape(&self) -> PointN<N> {
        self.builder().chunk_shape()
    }

    #[inline]
    pub fn ambient_value(&self) -> T {
        self.builder().ambient_value()
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    Store: ChunkReadStorage<N, Bldr::Chunk>,
{
    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_chunk(&self, key: ChunkKey<N>) -> Option<&Bldr::Chunk> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.get(key)
    }

    /// Get the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn clone_point(&self, lod: u8, p: PointN<N>) -> T
    where
        T: Clone,
        <Bldr::Chunk as Chunk>::Array: Get<PointN<N>, Item = T>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);

        self.get_chunk(ChunkKey::new(lod, chunk_min))
            .map(|chunk| chunk.array().get(p))
            .unwrap_or_else(|| self.ambient_value.clone())
    }

    /// Get a reference to the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn get_point<'a, Ref>(&'a self, lod: u8, p: PointN<N>) -> Ref
    where
        <Bldr::Chunk as Chunk>::Array: GetRef<'a, PointN<N>, Item = Ref>,
        Ref: MultiRef<'a, Data = T>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);

        self.get_chunk(ChunkKey::new(lod, chunk_min))
            .map(|chunk| chunk.array().get_ref(p))
            .unwrap_or_else(|| Ref::from_data_ref(&self.ambient_value))
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be represented by an `AmbientExtent`.
    #[inline]
    pub fn visit_chunks(
        &self,
        lod: u8,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(Either<&Bldr::Chunk, (&ExtentN<N>, AmbientExtent<N, T>)>),
    ) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(ChunkKey::new(lod, chunk_min)) {
                visitor(Either::Left(chunk))
            } else {
                let chunk_extent = self.indexer.extent_for_chunk_with_min(chunk_min);
                visitor(Either::Right((
                    &chunk_extent,
                    AmbientExtent::new(self.builder.ambient_value()),
                )))
            }
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_chunks(
        &self,
        lod: u8,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&Bldr::Chunk),
    ) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(ChunkKey::new(lod, chunk_min)) {
                visitor(chunk)
            }
        }
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    Store: ChunkWriteStorage<N, Bldr::Chunk>,
{
    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn write_chunk(&mut self, key: ChunkKey<N>, chunk: Bldr::Chunk) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.write(key, chunk);
    }

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn replace_chunk(&mut self, key: ChunkKey<N>, chunk: Bldr::Chunk) -> Option<Bldr::Chunk> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.replace(key, chunk)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk(&mut self, key: ChunkKey<N>) -> Option<&mut Bldr::Chunk> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.get_mut(key)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, `create_chunk` is called to insert one.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Bldr::Chunk,
    ) -> &mut Bldr::Chunk {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.get_mut_or_insert_with(key, create_chunk)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, a new chunk is created with the ambient value.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_ambient(&mut self, key: ChunkKey<N>) -> &mut Bldr::Chunk {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        let Self {
            indexer,
            storage,
            builder,
            ..
        } = self;
        let chunk_min = key.minimum;

        storage.get_mut_or_insert_with(key, || {
            builder.new_ambient(indexer.extent_for_chunk_with_min(chunk_min))
        })
    }

    /// Get a mutable reference to the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn get_mut_point<'a, Mut>(&'a mut self, lod: u8, p: PointN<N>) -> Mut
    where
        <Bldr::Chunk as Chunk>::Array: GetMut<'a, PointN<N>, Item = Mut>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);
        let chunk = self.get_mut_chunk_or_insert_ambient(ChunkKey::new(lod, chunk_min));

        chunk.array_mut().get_mut(p)
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be created first with ambient value.
    #[inline]
    pub fn visit_mut_chunks(
        &mut self,
        lod: u8,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Bldr::Chunk),
    ) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            visitor(self.get_mut_chunk_or_insert_ambient(ChunkKey::new(lod, chunk_min)));
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_mut_chunks(
        &mut self,
        lod: u8,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Bldr::Chunk),
    ) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            if let Some(chunk) = self.get_mut_chunk(ChunkKey::new(lod, chunk_min)) {
                visitor(chunk)
            }
        }
    }

    #[inline]
    pub fn delete_chunk(&mut self, key: ChunkKey<N>) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));
        self.storage.delete(key);
    }

    #[inline]
    pub fn pop_chunk(&mut self, key: ChunkKey<N>) -> Option<Bldr::Chunk> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.storage.pop(key)
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    for<'r> ChunkMapLodView<&'r mut Self>: FillExtent<N, Item = T>,
{
    /// Fill all of `extent` with the same `value`.
    #[inline]
    pub fn fill_extent(&mut self, lod: u8, extent: &ExtentN<N>, value: T) {
        self.lod_view_mut(lod).fill_extent(extent, value)
    }
}

impl<'a, N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: IterChunkKeys<'a, N>,
{
    /// The smallest extent that bounds all chunks in level of detail `lod`.
    pub fn bounding_extent(&'a self, lod: u8) -> ExtentN<N> {
        bounding_extent(
            self.storage
                .chunk_keys()
                .filter(|key| key.lod == lod)
                .flat_map(|key| {
                    let chunk_extent = self.indexer.extent_for_chunk_with_min(key.minimum);

                    vec![chunk_extent.minimum, chunk_extent.max()].into_iter()
                }),
        )
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

impl<N, T> ForEach<N, PointN<N>> for AmbientExtent<N, T>
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

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{access_traits::*, Array3x1};

    use building_blocks_core::prelude::*;

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);
    const BUILDER: ChunkMapBuilder3x1<i32> = ChunkMapBuilder3x1::new(CHUNK_SHAPE, 0);

    #[test]
    fn write_and_read_points() {
        let mut map = BUILDER.build_with_hash_map_storage();

        let mut lod0 = map.lod_view_mut(0);

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
            assert_eq!(lod0.get_mut(PointN(p)), &mut 0);
            *lod0.get_mut(PointN(p)) = 1;
            assert_eq!(lod0.get_mut(PointN(p)), &mut 1);
        }
    }

    #[test]
    fn write_extent_with_for_each_then_read() {
        let mut map = BUILDER.build_with_hash_map_storage();

        let mut lod0 = map.lod_view_mut(0);

        let write_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        lod0.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if write_extent.contains(p) {
                assert_eq!(lod0.get(p), 1);
            } else {
                assert_eq!(lod0.get(p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        let array = Array3x1::fill(extent_to_copy, 1);

        let mut map = BUILDER.build_with_hash_map_storage();

        let mut lod0 = map.lod_view_mut(0);

        copy_extent(&extent_to_copy, &array, &mut lod0);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(p) {
                assert_eq!(lod0.get(p), 1);
            } else {
                assert_eq!(lod0.get(p), 0);
            }
        }
    }

    #[test]
    fn multichannel_accessors() {
        let builder = ChunkMapBuilder3x2::new(CHUNK_SHAPE, (0, 'a'));
        let mut map = builder.build_with_hash_map_storage();

        let mut lod0 = map.lod_view_mut(0);

        assert_eq!(lod0.get(Point3i::fill(1)), (0, 'a'));
        assert_eq!(lod0.get_ref(Point3i::fill(1)), (&0, &'a'));
        assert_eq!(lod0.get_mut(Point3i::fill(1)), (&mut 0, &mut 'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        lod0.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = 'b';
        });

        lod0.for_each(&extent, |_p, (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, 'b');
        });

        map.fill_extent(0, &extent, (1, 'b'));
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn multichannel_compressed_accessors() {
        use crate::{FastCompressibleChunkStorageNx2, LocalChunkCache, Lz4};

        let builder = ChunkMapBuilder3x2::new(CHUNK_SHAPE, (0, 'a'));
        let mut map = builder.build_with_write_storage(
            FastCompressibleChunkStorageNx2::with_bytes_compression(Lz4 { level: 10 }),
        );

        let mut lod0 = map.lod_view_mut(0);

        assert_eq!(lod0.get_mut(Point3i::fill(1)), (&mut 0, &mut 'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        lod0.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = 'b';
        });

        let local_cache = LocalChunkCache::new();
        let reader = map.reader(&local_cache);
        let lod0 = reader.lod_view(0);
        assert_eq!(lod0.get(Point3i::fill(1)), (0, 'a'));
        assert_eq!(lod0.get_ref(Point3i::fill(1)), (&0, &'a'));

        lod0.for_each(&extent, |_p, (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, 'b');
        });
    }
}
