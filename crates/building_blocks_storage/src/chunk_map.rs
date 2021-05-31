//! A sparse lattice map made of up array chunks. Designed for random access. Supports multiple levels of detail.
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
//! let mut map_view = map.lod_view_mut(0);
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *map_view.get_mut(p) = 1;
//! }
//!
//! // Even though the map is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks in LOD0.
//! let bounding_extent = map.bounding_extent(0);
//!
//! // Now we can read back the values.
//! let map_view = map.lod_view(0);
//! map_view.for_each(&bounding_extent, |p, value| {
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
//!     assert_eq!(map_view.get(p), 1);
//! }
//! assert_eq!(map_view.get(Point3i::fill(1)), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(32));
//! let mut dense_map = Array3x1::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &map_view, &mut dense_map);
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
//! let mut map_view = map.lod_view_mut(0);
//!
//! // You can write voxels the same as any other `ChunkMap`. As chunks are created, they will be placed in an LRU cache.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *map_view.get_mut(p) = 1;
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

use crate::{
    Array, ArrayCopySrc, Channel, ChunkHashMap, ChunkIndexer, ChunkKey, ChunkReadStorage,
    ChunkWriteStorage, FillChannels, ForEach, ForEachMut, ForEachMutPtr, Get, GetMut, GetRef,
    IntoMultiMut, IterChunkKeys, MultiMutPtr, MultiRef, ReadExtent, SmallKeyHashMap, WriteExtent,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerPoint, PointN};

use core::hash::Hash;
use either::Either;
use std::ops::{Deref, DerefMut};

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

/// A lattice map made up of same-shaped `Array` chunks. It takes a value at every possible `PointN`, because accesses made
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

/// An object that knows how to construct chunks for a `ChunkMap`.
pub trait ChunkMapBuilder<N, T>: Sized {
    type Chunk: Chunk;

    fn chunk_shape(&self) -> PointN<N>;

    fn ambient_value(&self) -> T;

    /// Construct a new chunk with entirely ambient values.
    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk;

    /// Create a new `ChunkMap` with the given `storage` which must implement both `ChunkReadStorage` and `ChunkWriteStorage`.
    fn build_with_rw_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Self, Store>
    where
        PointN<N>: IntegerPoint<N>,
        Store: ChunkReadStorage<N, Self::Chunk> + ChunkWriteStorage<N, Self::Chunk>,
    {
        ChunkMap::new(self, storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkReadStorage`.
    fn build_with_read_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Self, Store>
    where
        PointN<N>: IntegerPoint<N>,
        Store: ChunkReadStorage<N, Self::Chunk>,
    {
        ChunkMap::new(self, storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkWriteStorage`.
    fn build_with_write_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Self, Store>
    where
        PointN<N>: IntegerPoint<N>,
        Store: ChunkWriteStorage<N, Self::Chunk>,
    {
        ChunkMap::new(self, storage)
    }

    /// Create a new `ChunkMap` using a `SmallKeyHashMap` as the chunk storage.
    fn build_with_hash_map_storage(self) -> ChunkHashMap<N, T, Self>
    where
        PointN<N>: IntegerPoint<N>,
        ChunkKey<N>: Eq + Hash,
    {
        Self::build_with_rw_storage(self, SmallKeyHashMap::default())
    }
}

/// A `ChunkMapBuilder` for `Array` chunks.
#[derive(Clone, Copy)]
pub struct ChunkMapBuilderNxM<N, T, Chan> {
    pub chunk_shape: PointN<N>,
    pub ambient_value: T,
    marker: std::marker::PhantomData<Chan>,
}

impl<N, T, Chan> ChunkMapBuilderNxM<N, T, Chan> {
    pub const fn new(chunk_shape: PointN<N>, ambient_value: T) -> Self {
        Self {
            chunk_shape,
            ambient_value,
            marker: std::marker::PhantomData,
        }
    }
}

macro_rules! builder_type_alias {
    ($name:ident, $dim:ty, $( $chan:ident ),+ ) => {
        pub type $name<$( $chan ),+> = ChunkMapBuilderNxM<$dim, ($($chan),+), ($(Channel<$chan>),+)>;
    };
}

pub mod multichannel_aliases {
    use super::*;

    /// A `ChunkMapBuilder` for `ArrayNx1` chunks.
    pub type ChunkMapBuilderNx1<N, A> = ChunkMapBuilderNxM<N, A, Channel<A>>;

    /// A `ChunkMapBuilder` for `Array2x1` chunks.
    pub type ChunkMapBuilder2x1<A> = ChunkMapBuilderNxM<[i32; 2], A, Channel<A>>;
    builder_type_alias!(ChunkMapBuilder2x2, [i32; 2], A, B);
    builder_type_alias!(ChunkMapBuilder2x3, [i32; 2], A, B, C);
    builder_type_alias!(ChunkMapBuilder2x4, [i32; 2], A, B, C, D);
    builder_type_alias!(ChunkMapBuilder2x5, [i32; 2], A, B, C, D, E);
    builder_type_alias!(ChunkMapBuilder2x6, [i32; 2], A, B, C, D, E, F);

    /// A `ChunkMapBuilder` for `Array3x1` chunks.
    pub type ChunkMapBuilder3x1<A> = ChunkMapBuilderNxM<[i32; 3], A, Channel<A>>;
    builder_type_alias!(ChunkMapBuilder3x2, [i32; 3], A, B);
    builder_type_alias!(ChunkMapBuilder3x3, [i32; 3], A, B, C);
    builder_type_alias!(ChunkMapBuilder3x4, [i32; 3], A, B, C, D);
    builder_type_alias!(ChunkMapBuilder3x5, [i32; 3], A, B, C, D, E);
    builder_type_alias!(ChunkMapBuilder3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;

impl<N, T, Chan> ChunkMapBuilder<N, T> for ChunkMapBuilderNxM<N, T, Chan>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Chan: FillChannels<Data = T>,
{
    type Chunk = Array<N, Chan>;

    fn chunk_shape(&self) -> PointN<N> {
        self.chunk_shape
    }

    fn ambient_value(&self) -> T {
        self.ambient_value.clone()
    }

    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk {
        Array::fill(extent, self.ambient_value())
    }
}

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
    pub fn lod_view<'a>(&'a self, lod: u8) -> ChunkMapLodView<&'a Self> {
        ChunkMapLodView {
            delegate: self,
            lod,
        }
    }

    /// Get a mutable view of a single level of detail `lod` in order to use the access traits.
    #[inline]
    pub fn lod_view_mut<'a>(&'a mut self, lod: u8) -> ChunkMapLodView<&'a mut Self> {
        ChunkMapLodView {
            delegate: self,
            lod,
        }
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
        let chunk_min = key.minimum.clone();

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

impl<N, T, Bldr, Store, MutPtr> ChunkMap<N, T, Bldr, Store>
where
    for<'r> ChunkMapLodView<&'r mut Self>: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    T: Clone,
    MutPtr: MultiMutPtr<Data = T>,
{
    /// Fill all of `extent` with the same `value`.
    #[inline]
    pub fn fill_extent(&mut self, lod: u8, extent: &ExtentN<N>, value: T) {
        let mut view = self.lod_view_mut(lod);
        // PERF: write whole chunks using a fast path
        unsafe {
            view.for_each_mut_ptr(extent, |_p, ptr| ptr.write(value.clone()));
        }
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

/// A view of a single level of detail in a `ChunkMap` for the unambiguous implementation of access traits.
pub struct ChunkMapLodView<Delegate> {
    delegate: Delegate,
    lod: u8,
}

impl<Delegate> ChunkMapLodView<Delegate> {
    #[inline]
    pub fn lod(&self) -> u8 {
        self.lod
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, Delegate, N, T, Bldr, Store> Get<PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: Get<PointN<N>, Item = T>,
    Store: ChunkReadStorage<N, Bldr::Chunk>,
{
    type Item = T;

    #[inline]
    fn get(&self, p: PointN<N>) -> Self::Item {
        self.delegate.clone_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Bldr, Store, Ref> GetRef<'a, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: GetRef<'a, PointN<N>, Item = Ref>,
    Store: 'a + ChunkReadStorage<N, Bldr::Chunk>,
    Ref: MultiRef<'a, Data = T>,
{
    type Item = Ref;

    #[inline]
    fn get_ref(&'a self, p: PointN<N>) -> Self::Item {
        self.delegate.get_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Bldr, Store, Mut> GetMut<'a, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: GetMut<'a, PointN<N>, Item = Mut>,
    Store: 'a + ChunkWriteStorage<N, Bldr::Chunk>,
{
    type Item = Mut;

    #[inline]
    fn get_mut(&'a mut self, p: PointN<N>) -> Self::Item {
        self.delegate.get_mut_point(self.lod, p)
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<Delegate, N, T, Bldr, Store> ForEach<N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: ForEach<N, PointN<N>, Item = T>,
    T: Clone,
    Store: ChunkReadStorage<N, Bldr::Chunk>,
{
    type Item = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.delegate
            .visit_chunks(self.lod, extent, |chunk| match chunk {
                Either::Left(chunk) => {
                    chunk.array().for_each(extent, |p, value| f(p, value));
                }
                Either::Right((chunk_extent, ambient)) => {
                    ambient.for_each(&extent.intersection(&chunk_extent), |p, value| f(p, value))
                }
            });
    }
}

impl<Delegate, N, T, Bldr, Store, MutPtr> ForEachMutPtr<N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    Store: ChunkWriteStorage<N, Bldr::Chunk>,
{
    type Item = MutPtr;

    #[inline]
    unsafe fn for_each_mut_ptr(
        &mut self,
        extent: &ExtentN<N>,
        mut f: impl FnMut(PointN<N>, Self::Item),
    ) {
        self.delegate.visit_mut_chunks(self.lod, extent, |chunk| {
            chunk
                .array_mut()
                .for_each_mut_ptr(extent, |p, ptr| f(p, ptr))
        });
    }
}

impl<'a, Delegate, N, Mut, MutPtr> ForEachMut<'a, N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Self: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    MutPtr: IntoMultiMut<'a, MultiMut = Mut>,
{
    type Item = Mut;

    #[inline]
    fn for_each_mut(&'a mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        unsafe { self.for_each_mut_ptr(extent, |p, ptr| f(p, ptr.into_multi_mut())) }
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, Delegate, N, T, Bldr, Store> ReadExtent<'a, N> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    T: 'a + Clone,
    Store: 'a + ChunkReadStorage<N, Bldr::Chunk>,
{
    type Src = ChunkCopySrc<N, T, &'a Bldr::Chunk>;
    type SrcIter = ChunkCopySrcIter<N, T, &'a Bldr::Chunk>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .delegate
            .indexer
            .chunk_mins_for_extent(extent)
            .map(|chunk_min| {
                let chunk_extent = self.delegate.indexer.extent_for_chunk_with_min(chunk_min);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.delegate
                        .get_chunk(ChunkKey::new(self.lod, chunk_min))
                        .map(|chunk| Either::Left(ArrayCopySrc(chunk)))
                        .unwrap_or_else(|| {
                            Either::Right(AmbientExtent::new(self.delegate.builder.ambient_value()))
                        }),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If `Array` supports writing from type Src, then so does ChunkMap.
impl<Delegate, N, T, Bldr, Store, Src> WriteExtent<N, Src> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: WriteExtent<N, Src>,
    Store: ChunkWriteStorage<N, Bldr::Chunk>,
    Src: Clone,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        self.delegate.visit_mut_chunks(self.lod, extent, |chunk| {
            chunk.array_mut().write_extent(extent, src.clone())
        });
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

    use crate::{copy_extent, Array3x1, Get};

    use building_blocks_core::prelude::*;

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);
    const BUILDER: ChunkMapBuilder3x1<i32> = ChunkMapBuilder3x1::new(CHUNK_SHAPE, 0);

    #[test]
    fn write_and_read_points() {
        let mut map = BUILDER.build_with_hash_map_storage();

        let mut view = map.lod_view_mut(0);

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
            assert_eq!(view.get_mut(PointN(p)), &mut 0);
            *view.get_mut(PointN(p)) = 1;
            assert_eq!(view.get_mut(PointN(p)), &mut 1);
        }
    }

    #[test]
    fn write_extent_with_for_each_then_read() {
        let mut map = BUILDER.build_with_hash_map_storage();

        let mut view = map.lod_view_mut(0);

        let write_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        view.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if write_extent.contains(p) {
                assert_eq!(view.get(p), 1);
            } else {
                assert_eq!(view.get(p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));
        let array = Array3x1::fill(extent_to_copy, 1);

        let mut map = BUILDER.build_with_hash_map_storage();

        let mut view = map.lod_view_mut(0);

        copy_extent(&extent_to_copy, &array, &mut view);

        let read_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(p) {
                assert_eq!(view.get(p), 1);
            } else {
                assert_eq!(view.get(p), 0);
            }
        }
    }

    #[test]
    fn multichannel_accessors() {
        let builder = ChunkMapBuilder3x2::new(CHUNK_SHAPE, (0, 'a'));
        let mut map = builder.build_with_hash_map_storage();

        let mut view = map.lod_view_mut(0);

        assert_eq!(view.get(Point3i::fill(1)), (0, 'a'));
        assert_eq!(view.get_ref(Point3i::fill(1)), (&0, &'a'));
        assert_eq!(view.get_mut(Point3i::fill(1)), (&mut 0, &mut 'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        view.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = 'b';
        });

        view.for_each(&extent, |_p, (num, letter)| {
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

        let mut view = map.lod_view_mut(0);

        assert_eq!(view.get_mut(Point3i::fill(1)), (&mut 0, &mut 'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        view.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = 'b';
        });

        let local_cache = LocalChunkCache::new();
        let reader = map.reader(&local_cache);
        let view = reader.lod_view(0);
        assert_eq!(view.get(Point3i::fill(1)), (0, 'a'));
        assert_eq!(view.get_ref(Point3i::fill(1)), (&0, &'a'));

        view.for_each(&extent, |_p, (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, 'b');
        });
    }
}
