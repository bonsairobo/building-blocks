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
//! # Chunk Storage
//!
//! `ChunkMap<N, T, Bldr, Store>` depends on a backing chunk storage `Store`, which can implement some of `ChunkReadStorage` or
//! `ChunkWriteStorage`. A storage can be as simple as a `HashMap`, which provides good performance for both iteration and
//! random access. It could also be something more memory efficient like `FastCompressibleChunkStorage` which perform nearly as
//! well but involves some overhead for caching and compression.
//!
//! # Serialization
//!
//! While `ChunkMap` derives `Deserialize` and `Serialize`, it will only be serializable if its constituent types are
//! serializable. You should expect a `ChunkHashMap` with simple `Array` chunks to be serializable, but a `CompressibleChunkMap`
//! is *not*.
//!
//! However using `serde` for serializing large dynamic chunk maps is discouraged. Instead there is a `ChunkDb` backed by the
//! `sled` embedded database which supports transactions and compression.
//!
//! # Example `ChunkHashMap` Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let chunk_shape = Point3i::fill(16);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(ChunkMapConfig { chunk_shape, ambient_value, root_lod: 0 });
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
//! let builder = ChunkMapBuilder3x1::new(ChunkMapConfig { chunk_shape, ambient_value, root_lod: 0 });
//! let mut map = builder.build_with_storage(
//!     || FastCompressibleChunkStorageNx1::with_bytes_compression(Lz4 { level: 10 })
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
//! map.lod_storage_mut(0).compress_lru();
//!
//! let bounding_extent = map.bounding_extent(0);
//! map.lod_view(0).for_each(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, 1);
//!     } else {
//!         assert_eq!(value, 0);
//!     }
//! });
//!
//! // For efficient caching, you should occasionally flush your local caches back into the global cache.
//! map.lod_storage_mut(0).flush_thread_local_caches();
//! ```

pub mod builder;
pub mod clipmap;
pub mod lod_view;
pub mod sampling;

pub use builder::*;
pub use clipmap::*;
pub use lod_view::*;
pub use sampling::*;

use crate::{
    chunk::ChunkIndexer,
    dev_prelude::{
        Array, ChunkStorage, FillExtent, ForEach, GetMutUnchecked, GetRefUnchecked, GetUnchecked,
        IterChunkKeys,
    },
    multi_ptr::MultiRef,
};

use building_blocks_core::{
    bounding_extent,
    point_traits::{IntegerPoint, Neighborhoods},
    ExtentN, PointN,
};

use either::Either;
use serde::{Deserialize, Serialize};

/// The key for a chunk at a particular level of detail.
#[allow(clippy::derive_hash_xor_eq)] // This is fine, the custom PartialEq is the same as what would've been derived.
#[derive(Debug, Deserialize, Hash, Eq, Serialize)]
pub struct ChunkKey<N> {
    /// The minimum point of the chunk.
    pub minimum: PointN<N>,
    /// The level of detail. From highest resolution at 0 to lowest resolution at MAX_LOD.
    pub lod: u8,
}

// A few of these traits could be derived. But it seems that derive will not help the compiler infer trait bounds as well.

impl<N> Clone for ChunkKey<N>
where
    PointN<N>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            minimum: self.minimum.clone(),
            lod: self.lod,
        }
    }
}
impl<N> Copy for ChunkKey<N> where PointN<N>: Copy {}

impl<N> PartialEq for ChunkKey<N>
where
    PointN<N>: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.minimum == other.minimum && self.lod == other.lod
    }
}

/// A 2-dimensional `ChunkKey`.
pub type ChunkKey2 = ChunkKey<[i32; 2]>;
/// A 3-dimensional `ChunkKey`.
pub type ChunkKey3 = ChunkKey<[i32; 3]>;

impl<N> ChunkKey<N> {
    pub fn new(lod: u8, chunk_minimum: PointN<N>) -> Self {
        Self {
            lod,
            minimum: chunk_minimum,
        }
    }
}

/// The user-accessible data stored in each chunk of a `ChunkMap`.
///
/// This crate provides a blanket impl for any `Array`, but users can also provide an impl that affords further customization.
pub trait UserChunk {
    /// The inner array type. This makes it easier for `UserChunk` implementations to satisfy access trait bounds by inheriting
    /// them from existing `Array` types.
    type Array;

    /// Borrow the inner array.
    fn array(&self) -> &Self::Array;

    /// Mutably borrow the inner array.
    fn array_mut(&mut self) -> &mut Self::Array;
}

impl<N, Chan> UserChunk for Array<N, Chan> {
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

/// The container of a `U: UserChunk` that's actually stored in a chunk storage.
pub struct ChunkNode<U> {
    /// Parent chunks are `None` until written or downsampled into. This means that users can opt-in to storing downsampled
    /// chunks, which requires more memory.
    pub user_chunk: Option<U>,
    child_mask: u8,
}

impl<U> ChunkNode<U> {
    #[inline]
    pub(crate) fn new_empty() -> Self {
        Self {
            user_chunk: None,
            child_mask: 0,
        }
    }

    #[inline]
    pub(crate) fn new(user_chunk: Option<U>, child_mask: u8) -> Self {
        Self {
            user_chunk,
            child_mask,
        }
    }

    #[inline]
    pub(crate) fn child_mask(&self) -> u8 {
        self.child_mask
    }

    fn has_child(&self, corner_index: u8) -> bool {
        child_mask_has_child(self.child_mask, corner_index)
    }
}

#[inline]
pub(crate) fn child_mask_has_child(mask: u8, corner_index: u8) -> bool {
    mask & (1 << corner_index) != 0
}

/// A lattice map made up of same-shaped [Array] chunks. For each level of detail, it takes a value at every possible
/// [`PointN`], because accesses made outside of the stored chunks will return some ambient value specified on creation.
///
/// [`ChunkMap`] is generic over the type used to actually store the [`UserChunk`]s. You can use any storage that implements
/// [`ChunkReadStorage`] or [`ChunkWriteStorage`]. Being a lattice map, [`ChunkMapLodView`] will implement various access
/// traits, depending on the capabilities of the chunk storage.
///
/// If the chunk storage implements [`ChunkReadStorage`], then [`ChunkMapLodView`] will implement:
/// - [`Get`](crate::access_traits::Get)
/// - [`ForEach`](crate::access_traits::ForEach)
/// - [`ReadExtent`](crate::access_traits::ReadExtent)
///
/// If the chunk storage implements [`ChunkWriteStorage`], then [`ChunkMapLodView`] will implement:
/// - [`GetMut`](crate::access_traits::GetMut)
/// - [`ForEachMut`](crate::access_traits::ForEachMut)
/// - [`WriteExtent`](crate::access_traits::WriteExtent)
#[derive(Deserialize, Serialize)]
pub struct ChunkMap<N, T, Bldr, Store> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,
    storages: Vec<Store>,
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
    T: Clone,
    Bldr: ChunkMapBuilder<N, T>,
{
    /// Creates a map using the given `storages`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    fn new(builder: Bldr, storages: Vec<Store>) -> Self {
        let indexer = ChunkIndexer::new(builder.chunk_shape());
        let ambient_value = builder.ambient_value();

        Self {
            indexer,
            storages,
            builder,
            ambient_value,
        }
    }

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
    Bldr: ChunkMapBuilder<N, T>,
{
    #[inline]
    pub fn root_lod(&self) -> u8 {
        self.builder().root_lod()
    }
}

impl<N, T, Bldr, Store> ChunkMap<N, T, Bldr, Store> {
    /// Consumes `self` and returns the backing chunk storage.
    #[inline]
    pub fn take_storages(self) -> Vec<Store> {
        self.storages
    }

    /// Borrows the internal chunk storages for all LODs.
    #[inline]
    pub fn storages(&self) -> &[Store] {
        &self.storages
    }

    /// Mutably borrows the internal chunk storages for all LODs.
    #[inline]
    pub fn storages_mut(&mut self) -> &mut [Store] {
        &mut self.storages
    }

    /// Borrows the internal chunk storage for `lod`.
    #[inline]
    pub fn lod_storage(&self, lod: u8) -> &Store {
        &self.storages[lod as usize]
    }

    /// Mutably borrows the internal chunk storage for `lod`.
    #[inline]
    pub fn lod_storage_mut(&mut self, lod: u8) -> &mut Store {
        &mut self.storages[lod as usize]
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

impl<N, T, Usr, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    fn get_chunk_node(&self, key: ChunkKey<N>) -> Option<&ChunkNode<Usr>> {
        self.lod_storage(key.lod).get(key.minimum)
    }

    fn get_mut_chunk_node(&mut self, key: ChunkKey<N>) -> Option<&mut ChunkNode<Usr>> {
        self.lod_storage_mut(key.lod).get_mut(key.minimum)
    }

    fn write_chunk_node(&mut self, key: ChunkKey<N>, node: ChunkNode<Usr>) {
        self.lod_storage_mut(key.lod).write(key.minimum, node)
    }

    fn delete_chunk_node(&mut self, key: ChunkKey<N>) {
        self.lod_storage_mut(key.lod).delete(key.minimum)
    }

    fn pop_chunk_node(&mut self, key: ChunkKey<N>) -> Option<ChunkNode<Usr>> {
        self.lod_storage_mut(key.lod).pop(key.minimum)
    }
}

impl<N, T, Usr, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_chunk(&self, key: ChunkKey<N>) -> Option<&Usr> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.get_chunk_node(key)
            .map(|ch| ch.user_chunk.as_ref())
            .flatten()
    }

    /// Get the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn clone_point(&self, lod: u8, p: PointN<N>) -> T
    where
        T: Clone,
        Usr::Array: GetUnchecked<PointN<N>, Item = T>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);

        self.get_chunk(ChunkKey::new(lod, chunk_min))
            .map(|chunk| unsafe { chunk.array().get_unchecked(p) })
            .unwrap_or_else(|| self.ambient_value.clone())
    }

    /// Get a reference to the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn get_point<'a, Ref>(&'a self, lod: u8, p: PointN<N>) -> Ref
    where
        Usr: 'a,
        Usr::Array: GetRefUnchecked<'a, PointN<N>, Item = Ref>,
        Ref: MultiRef<'a, Data = T>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);

        self.get_chunk(ChunkKey::new(lod, chunk_min))
            .map(|chunk| unsafe { chunk.array().get_ref_unchecked(p) })
            .unwrap_or_else(|| Ref::from_data_ref(&self.ambient_value))
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be represented by an `AmbientExtent`.
    #[inline]
    pub fn visit_chunks(
        &self,
        lod: u8,
        extent: ExtentN<N>,
        mut visitor: impl FnMut(Either<&Usr, (&ExtentN<N>, AmbientExtent<N, T>)>),
    ) where
        T: Clone,
    {
        // PERF: we could traverse the octree to avoid using hashing to check for occupancy
        for chunk_min in self.indexer.chunk_mins_for_extent(&extent) {
            if let Some(chunk) = self.get_chunk(ChunkKey::new(lod, chunk_min)) {
                visitor(Either::Left(chunk))
            } else {
                let chunk_extent = self.indexer.extent_for_chunk_with_min(chunk_min);
                visitor(Either::Right((
                    &chunk_extent,
                    AmbientExtent::new(self.ambient_value.clone()),
                )))
            }
        }
    }
}

impl<N, T, Usr, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Bldr: ChunkMapBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    /// Call `visitor` on all occupied chunks that overlap `extent` in level `lod`.
    #[inline]
    pub fn visit_occupied_chunks(
        &self,
        lod: u8,
        extent: ExtentN<N>,
        mut visitor: impl FnMut(&Usr),
    ) {
        let root_lod = self.root_lod();
        assert!(lod <= root_lod);

        // Get an extent at each level that covers all ancestors we want to visit.
        let covering_extents = self
            .indexer
            .covering_ancestor_extents_for_lods(root_lod, lod, extent);

        for root_chunk_min in self
            .indexer
            .chunk_mins_for_extent(&covering_extents[root_lod as usize])
        {
            self.visit_occupied_chunks_recursive(
                ChunkKey::new(root_lod, root_chunk_min),
                lod,
                &covering_extents,
                &mut visitor,
            );
        }
    }

    fn visit_occupied_chunks_recursive(
        &self,
        node_key: ChunkKey<N>,
        lod: u8,
        covering_extents: &[ExtentN<N>],
        visitor: &mut impl FnMut(&Usr),
    ) {
        if let Some(node) = self.get_chunk_node(node_key) {
            if node_key.lod == lod {
                if let Some(chunk) = &node.user_chunk {
                    visitor(chunk);
                }
                return;
            }

            let next_level_covering_extent = &covering_extents[node_key.lod as usize - 1];
            for child_i in 0..PointN::NUM_CORNERS {
                if node.has_child(child_i) {
                    let child_key = self.indexer.child_chunk_key(node_key, child_i);

                    // Skip if this node is not an ancestor.
                    let child_extent = self.indexer.extent_for_chunk_with_min(child_key.minimum);
                    if child_extent
                        .intersection(next_level_covering_extent)
                        .is_empty()
                    {
                        continue;
                    }

                    self.visit_occupied_chunks_recursive(child_key, lod, covering_extents, visitor)
                }
            }
        }
    }

    fn link_node(&mut self, mut key: ChunkKey<N>) {
        // PERF: when writing many chunks, we would revisit nodes many times as we travel up to the root for every chunk. We
        // might do better by inserting them in a batch and sorting the chunks in morton order before linking into the tree.
        while key.lod < self.root_lod() {
            let parent = self.indexer.parent_chunk_key(key);
            let mut parent_already_exists = true;
            let parent_node =
                self.storages[parent.lod as usize].get_mut_or_insert_with(parent.minimum, || {
                    parent_already_exists = false;
                    ChunkNode::new_empty()
                });
            let child_corner_index = self.indexer.corner_index(key.minimum);
            parent_node.child_mask |= 1 << child_corner_index;
            if parent_already_exists {
                return;
            }
            key = parent;
        }
    }

    fn unlink_node(&mut self, mut key: ChunkKey<N>) {
        // PERF: when removing many chunks, we would revisit nodes many times as we travel up to the root for every chunk. We
        // might do better by removing them in a batch and sorting the chunks in morton order before unlinking from the tree.
        while key.lod < self.root_lod() {
            let parent = self.indexer.parent_chunk_key(key);
            let parent_node = self.storages[parent.lod as usize]
                .get_mut(parent.minimum)
                .unwrap();
            let child_corner_index = self.indexer.corner_index(key.minimum);
            parent_node.child_mask &= !(1 << child_corner_index);
            if parent_node.child_mask != 0 || parent_node.user_chunk.is_some() {
                return;
            }
            key = parent;
        }
    }

    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn write_chunk(&mut self, key: ChunkKey<N>, chunk: Usr) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        let node = self
            .lod_storage_mut(key.lod)
            .get_mut_or_insert_with(key.minimum, ChunkNode::new_empty);
        node.user_chunk = Some(chunk);
        self.link_node(key);
    }

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn replace_chunk(&mut self, key: ChunkKey<N>, chunk: Usr) -> Option<Usr> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.link_node(key);

        let node = self
            .lod_storage_mut(key.lod)
            .get_mut_or_insert_with(key.minimum, ChunkNode::new_empty);
        node.user_chunk.replace(chunk)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk(&mut self, key: ChunkKey<N>) -> Option<&mut Usr> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.get_mut_chunk_node(key)
            .and_then(|c| c.user_chunk.as_mut())
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, `create_chunk` is called to insert one.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Usr,
    ) -> &mut Usr {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.link_node(key);

        self.lod_storage_mut(key.lod)
            .get_mut_or_insert_with(key.minimum, ChunkNode::new_empty)
            .user_chunk
            .get_or_insert_with(|| create_chunk())
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, a new chunk is created with the ambient value.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_ambient(&mut self, key: ChunkKey<N>) -> &mut Usr {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.link_node(key);

        let Self {
            indexer,
            storages,
            builder,
            ..
        } = self;
        let chunk_min = key.minimum;

        storages[key.lod as usize]
            .get_mut_or_insert_with(key.minimum, ChunkNode::new_empty)
            .user_chunk
            .get_or_insert_with(|| {
                builder.new_ambient(indexer.extent_for_chunk_with_min(chunk_min))
            })
    }

    /// Get a mutable reference to the values at point `p` in level of detail `lod`.
    #[inline]
    pub fn get_mut_point<'a, Mut>(&'a mut self, lod: u8, p: PointN<N>) -> Mut
    where
        Usr: 'a,
        Usr::Array: GetMutUnchecked<'a, PointN<N>, Item = Mut>,
    {
        let chunk_min = self.indexer.min_of_chunk_containing_point(p);
        let chunk = self.get_mut_chunk_or_insert_ambient(ChunkKey::new(lod, chunk_min));

        unsafe { chunk.array_mut().get_mut_unchecked(p) }
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be created first with ambient value.
    #[inline]
    pub fn visit_mut_chunks(
        &mut self,
        lod: u8,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Usr),
    ) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            visitor(self.get_mut_chunk_or_insert_ambient(ChunkKey::new(lod, chunk_min)));
        }
    }

    #[inline]
    pub fn delete_chunk(&mut self, key: ChunkKey<N>) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));
        self.unlink_node(key);
        self.delete_chunk_node(key);
    }

    #[inline]
    pub fn pop_chunk(&mut self, key: ChunkKey<N>) -> Option<Usr> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));
        self.unlink_node(key);
        self.pop_chunk_node(key).and_then(|c| c.user_chunk)
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
        bounding_extent(self.lod_storage(lod).chunk_keys().flat_map(|&chunk_min| {
            let chunk_extent = self.indexer.extent_for_chunk_with_min(chunk_min);

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

    use crate::prelude::*;

    use building_blocks_core::prelude::*;

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);
    const MAP_CONFIG: ChunkMapConfig<[i32; 3], i32> = ChunkMapConfig {
        chunk_shape: CHUNK_SHAPE,
        ambient_value: 0,
        root_lod: 2,
    };
    const MULTICHAN_MAP_CONFIG: ChunkMapConfig<[i32; 3], (i32, u8)> = ChunkMapConfig {
        chunk_shape: CHUNK_SHAPE,
        ambient_value: (0, b'a'),
        root_lod: 0,
    };

    #[test]
    fn write_and_read_points() {
        let mut map = ChunkMapBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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
        let mut map = ChunkMapBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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

        let mut map = ChunkMapBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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
    fn visit_occupied_chunks() {
        assert!(MAP_CONFIG.root_lod > 0);
        let mut map = ChunkMapBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

        let insert_chunk_mins = [PointN([0, 0, 0]), PointN([16, 0, 0]), PointN([32, 0, 0])];
        for chunk_min in insert_chunk_mins.iter() {
            let chunk_extent = map.indexer.extent_for_chunk_with_min(*chunk_min);
            map.write_chunk(
                ChunkKey::new(0, *chunk_min),
                Array3x1::fill(chunk_extent, 1),
            );
        }

        // This intentionally doesn't cover all of the inserted chunks, but it overlaps two roots.
        let visit_extent = Extent3i::from_min_and_shape(PointN([16, 0, 0]), Point3i::fill(17));

        let mut visited_lod0_chunk_mins = Vec::new();
        map.visit_occupied_chunks(0, visit_extent, |chunk| {
            visited_lod0_chunk_mins.push(chunk.extent().minimum)
        });

        assert_eq!(&visited_lod0_chunk_mins, &insert_chunk_mins[1..=2]);
    }

    #[test]
    fn multichannel_accessors() {
        let builder = ChunkMapBuilder3x2::new(MULTICHAN_MAP_CONFIG);
        let mut map = builder.build_with_hash_map_storage();

        let mut lod0 = map.lod_view_mut(0);

        assert_eq!(lod0.get(Point3i::fill(1)), (0, b'a'));
        assert_eq!(lod0.get_ref(Point3i::fill(1)), (&0, &b'a'));
        assert_eq!(lod0.get_mut(Point3i::fill(1)), (&mut 0, &mut b'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        lod0.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = b'b';
        });

        lod0.for_each(&extent, |_p, (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, b'b');
        });

        map.fill_extent(0, &extent, (1, b'b'));
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn multichannel_compressed_accessors() {
        use crate::prelude::{FastCompressibleChunkStorageNx2, Lz4};

        let builder = ChunkMapBuilder3x2::new(MULTICHAN_MAP_CONFIG);
        let mut map = builder.build_with_storage(|| {
            FastCompressibleChunkStorageNx2::with_bytes_compression(Lz4 { level: 10 })
        });

        let mut lod0 = map.lod_view_mut(0);

        assert_eq!(lod0.get_mut(Point3i::fill(1)), (&mut 0, &mut b'a'));

        let extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(80));

        lod0.for_each_mut(&extent, |_p, (num, letter)| {
            *num = 1;
            *letter = b'b';
        });

        let lod0 = map.lod_view(0);
        assert_eq!(lod0.get(Point3i::fill(1)), (0, b'a'));
        assert_eq!(lod0.get_ref(Point3i::fill(1)), (&0, &b'a'));

        lod0.for_each(&extent, |_p, (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, b'b');
        });
    }
}
