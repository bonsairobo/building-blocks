//! A hashed quadtree ([`ChunkTree2`]) or octree ([`ChunkTree3`]) of array chunks. Designed for both sparse iteration and random
//! access. Supports multiple levels of detail, downsampling, and clipmap algorithms.
//!
//! # Coordinates and Level of Detail
//!
//! A [`ChunkTree`] can have up to [`MAX_LODS`] levels of detail. Each level (LOD) can be thought of as a lattice map, i.e. a
//! map from [`PointN`] to arbitrary data `T`. Starting at LOD0, voxels have edge length 1. At LOD1, edge length 2. And at each
//! next level, the voxel edge length doubles. Equivalently, each next level halves the sample rate.
//!
//! Each level is also partitioned into *chunk slots*, all slots having the same shape (in voxels). Although all chunks have the
//! same "data shape," chunks at higher levels take up more "perceptual space." E.g. a chunk at LOD2 is 2^2 = 4 times as wide
//! when it is rendered as a chunk at LOD0.
//!
//! A chunk slot can either be occupied or vacant. Occupied slots store some type `U: UserChunk`, where users are free to
//! implement [`UserChunk`]. These chunks are densely populated via an [`Array`]. Vacant chunk slots do not take up any space,
//! and we assume that all points in such a slot take the same *ambient value*.
//!
//! # Iteration
//!
//! While the [`ChunkTree`] strikes a good balance between random access and iteration speed, you should always prefer iteration
//! when it is convenient; iteration amortizes the cost of hashing to find chunks, and the tree structure allows us to
//! efficiently skip over empty space.
//!
//! ## Extents
//!
//! You will commonly want to iterate over an [`ExtentN`] in a single level of detail. For this, you can construct a
//! [`ChunkTreeLodView`] and use the familiar access traits to iterate or copy data:
//!
//! - [`ForEach`](crate::access_traits::ForEach)
//! - [`ForEachMut`](crate::access_traits::ForEachMut)
//! - [`ReadExtent`](crate::access_traits::ReadExtent)
//! - [`WriteExtent`](crate::access_traits::WriteExtent)
//!
//! ## Tree Search
//!
//! The `ChunkTree` is technically a forest, because it has many roots. Since each LOD lives in a different `ChunkStorage`, we
//! are able to efficiently iterate over the full set of roots with `ChunkTree::visit_root_keys`. To continue the traversal, you
//! can use `ChunkTree::visit_child_keys`, or any of the other `visit_*` methods. Any kind of traversal can be implemented by
//! composing the key visitor methods.
//!
//! # Random Access
//!
//! ## Chunks
//!
//! While in a tree you normally expect a tree to require `O(log N)` time for random access, the [`ChunkTree`] only requires
//! `O(levels)` in the worst case for insertions and `O(1)` to get an existing chunk or check if it exists, since every chunk
//! ultimately lives in a hash map. And luckily, the hash map keys are very small and the
//! [`SmallKeyHashMap`](crate::SmallKeyHashMap) is very fast.
//!
//! When writing a chunk into a node that doesn't exist yet, the node will be linked to its nearest ancestor node, which may
//! need to create a new root in the worst case. All new linkage nodes will not contain any data until they are filled manually
//! or downsampled into.
//!
//! Chunks can be accessed directly by [`ChunkKey`] with the `get*_chunk*` methods. The key for a chunk is comprised of:
//!
//!   - the `u8` level of detail
//!   - the minimum [`PointN`] in the chunk, which is always a multiple of the chunk shape
//!
//! Chunk shape dimensions must be powers of 2, which allows for efficiently calculating a chunk minimum from any point in the
//! chunk.
//!
//! ## Points
//!
//! Being a lattice map, [`ChunkTreeLodView`] also implements these traits for random access of individual points:
//!
//! - [`Get`](crate::access_traits::Get)
//! - [`GetMut`](crate::access_traits::GetMut)
//!
//! But you should try to avoid using these if possible, since each call requires hashing to find a chunk that contains the
//! point you're looking for.
//!
//! # Downsampling
//!
//! Most of the time, you will just manipulate the data at LOD0, but if you need to downsample to save resources where coarser
//! resolution is acceptable, then you can use a [`ChunkDownsampler`] and the `ChunkTree::downsample_*` methods to populate
//! higher levels. Currently, two downsamplers are provided for you:
//!
//!   - [`PointDownsampler`]
//!   - [`SdfMeanDownsampler`]
//!
//! **NOTE**: If you want your downsampled data to have different number of channels than LOD0, then you will need to store the
//! downsampled chunks in a different [`ChunkTree`]. You will need to use the specialized
//! `ChunkTree::downsample_extent_into_self_with_lod0` method for this use case.
//!
//! # Chunk Storage
//!
//! The fully generic `ChunkTree<N, T, Bldr, Store>` depends on a backing chunk storage `Store`, which must implement the
//! [`ChunkStorage`] trait. A storage can be as simple as a [`SmallKeyHashMap`](crate::SmallKeyHashMap). It could also be
//! something more memory efficient like [`CompressibleChunkStorage`](self::CompressibleChunkStorage) which performs nearly as
//! well but involves some overhead for caching and compression.
//!
//! # Clipmap
//!
//! The `ChunkTree` is designed to be used as a clipmap, i.e. a structure that controls which chunks are visible and their level
//! of detail based on proximity to an observer (camera). Each [`NodeState`] in the tree knows:
//!
//!   - if there is chunk data that needs to be loaded
//!   - if the chunk is currently being rendered
//!
//! The `ChunkTree::clipmap_*` algorithms can be given a work budget and search the tree to find a set of chunks that need to be
//! loaded or rendered at a different level of detail. See the "lod terrain" example for proper usage.
//!
//! # Serialization
//!
//! While [`ChunkTree`] derives `Deserialize` and `Serialize`, it will only be serializable if its constituent types are
//! serializable. You should expect a [`HashMapChunkTree`] with simple [`Array`] chunks to be serializable, but a
//! [`CompressibleChunkTree`] is *not*. So you will not benefit from chunk compression when serializing with `serde`.
//!
//! While sometimes convenient, using `serde` for serializing large dynamic chunk maps is discouraged. Instead there is a
//! [`ChunkDb`](crate::database::ChunkDb) backed by the `sled` embedded database which supports transactions and compression.
//! You likely want to use that instead.
//!
//! # Example [`CompressibleChunkTree`] Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! // Make a tree with 8 levels of detail.
//! let ambient_value = 0;
//! let config = ChunkTreeConfig { chunk_shape: Point3i::fill(16), ambient_value, root_lod: 7 };
//! // Each LOD gets a separate storage (all of the same type), so we provide a factory closure for
//! // our storage. This storage supports compressing our least-recently-used chunks.
//! let storage_factory = || FastCompressibleChunkStorageNx1::with_bytes_compression(Lz4 { level: 10 });
//! // This particular builder knows how to construct our `Array3x1` chunks.
//! let mut tree = ChunkTreeBuilder3x1::new(config).build_with_storage(storage_factory);
//!
//! // We need to focus on a specific level of detail to use the access traits.
//! let mut lod0 = tree.lod_view_mut(0);
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted,
//! // since each point lands in a different chunk slot.
//! let write_points = [Point3i::fill(-100), Point3i::ZERO, Point3i::fill(100)];
//! for &p in write_points.iter() {
//!     *lod0.get_mut(p) = 1;
//! }
//! assert_eq!(tree.lod_storage(0).len_cached(), 3);
//!
//! // Even though the tree is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks in LOD0.
//! let bounding_extent = tree.lod_view(0).bounding_extent().unwrap();
//!
//! // Now we can read back the values.
//! let lod0 = tree.lod_view(0);
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
//! // Save some space by compressing the least recently used chunks. On further access to the
//! // compressed chunks, they will get decompressed and cached.
//! while !tree.lod_storage_mut(0).try_compress_lru() {}
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access inside some bounding extent. In this case it's most efficient to use
//! // an `Array`. If you only need access to one chunk, then you can just get it and use its `Array`
//! // directly. But if your query spans multiple chunks, you should copy the extent into a new `Array`.
//! let query_extent = Extent3i::from_min_and_shape(Point3i::fill(10), Point3i::fill(32));
//! let mut dense_map = Array3x1::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &tree.lod_view(0), &mut dense_map);
//!
//! // For efficient caching, you should occasionally flush your thread-local caches back into the main cache.
//! tree.lod_storage_mut(0).flush_thread_local_caches();
//! ```

pub mod builder;
pub mod clipmap;
pub mod indexer;
pub mod lod_view;
pub mod sampling;
pub mod storage;

pub use builder::*;
pub use clipmap::*;
pub use indexer::*;
pub use lod_view::*;
pub use sampling::*;
pub use storage::*;

use crate::{
    bitset::*,
    dev_prelude::{Array, ForEach, GetMutUnchecked, GetRefUnchecked, GetUnchecked},
    multi_ptr::MultiRef,
};

use building_blocks_core::{
    point_traits::{IntegerPoint, Neighborhoods},
    ExtentN, PointN,
};

use either::Either;
use serde::{Deserialize, Serialize};

/// The user-accessible data stored in each chunk of a [`ChunkTree`].
///
/// This crate provides a blanket impl for any `Array`, but users can also provide an impl that affords further customization.
/// If you implement your own `UserChunk`, you will also need to implement a corresponding [`ChunkTreeBuilder`].
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

/// A multiresolution lattice map made up of same-shaped [`Array`] chunks.
///
/// See the [module-level docs](self) for more info.
#[derive(Deserialize, Serialize)]
pub struct ChunkTree<N, T, Bldr, Store> {
    pub indexer: ChunkIndexer<N>,
    storages: Vec<Store>,
    builder: Bldr,
    ambient_value: T, // Needed for GetRef to return a reference to non-temporary value
}

/// A 2-dimensional `ChunkTree`.
pub type ChunkTree2<T, Bldr, Store> = ChunkTree<[i32; 2], T, Bldr, Store>;
/// A 3-dimensional `ChunkTree`.
pub type ChunkTree3<T, Bldr, Store> = ChunkTree<[i32; 3], T, Bldr, Store>;

/// An N-dimensional, single-channel `ChunkTree`.
pub type ChunkTreeNx1<N, T, Store> = ChunkTree<N, T, ChunkTreeBuilderNx1<N, T>, Store>;
/// A 2-dimensional, single-channel `ChunkTree`.
pub type ChunkTree2x1<T, Store> = ChunkTreeNx1<[i32; 2], T, Store>;
/// A 3-dimensional, single-channel `ChunkTree`.
pub type ChunkTree3x1<T, Store> = ChunkTreeNx1<[i32; 3], T, Store>;

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Bldr: ChunkTreeBuilder<N, T>,
{
    /// Creates a map using the given `storages`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    fn new(builder: Bldr, storages: Vec<Store>) -> Self {
        assert!((builder.root_lod() as usize) < MAX_LODS);

        let indexer = ChunkIndexer::new(builder.chunk_shape());
        let ambient_value = builder.ambient_value();

        Self {
            indexer,
            storages,
            builder,
            ambient_value,
        }
    }

    /// The value used for any point in a vacant chunk slot.
    #[inline]
    pub fn ambient_value(&self) -> T {
        self.builder().ambient_value()
    }

    /// Create and return a new chunk with entirely ambient values.
    pub fn new_ambient_chunk(&self, chunk_key: ChunkKey<N>) -> Bldr::Chunk {
        self.builder
            .new_ambient(self.indexer.extent_for_chunk_with_min(chunk_key.minimum))
    }
}

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    Bldr: ChunkTreeBuilder<N, T>,
{
    /// The LOD index for root nodes, i.e. the maximum LOD.
    #[inline]
    pub fn root_lod(&self) -> u8 {
        self.builder().root_lod()
    }
}

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: Clone,
    Bldr: ChunkTreeBuilder<N, T>,
{
    /// The data shape of any chunk, regardless of LOD.
    #[inline]
    pub fn chunk_shape(&self) -> PointN<N> {
        self.builder().chunk_shape()
    }
}

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store> {
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

    /// The `ChunkTreeBuilder` for this tree.
    #[inline]
    pub fn builder(&self) -> &Bldr {
        &self.builder
    }

    /// Get an immutable view of a single level of detail `lod` in order to use the access traits.
    #[inline]
    pub fn lod_view(&self, lod: u8) -> ChunkTreeLodView<&'_ Self> {
        ChunkTreeLodView {
            delegate: self,
            lod,
        }
    }

    /// Get a mutable view of a single level of detail `lod` in order to use the access traits.
    #[inline]
    pub fn lod_view_mut(&mut self, lod: u8) -> ChunkTreeLodView<&'_ mut Self> {
        ChunkTreeLodView {
            delegate: self,
            lod,
        }
    }
}

// Convenience adapters over the chunk storage.
impl<N, T, Usr, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
    /// Returns `true` iff the tree contains a chunk for `key`.
    #[inline]
    pub fn contains_chunk(&self, key: ChunkKey<N>) -> bool {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        if let Some((_, has_data)) = self.get_node_state(key) {
            has_data
        } else {
            false
        }
    }

    /// Borrows the `ChunkNode` for `key`.
    #[inline]
    pub fn get_node(&self, key: ChunkKey<N>) -> Option<&ChunkNode<Usr>> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage(key.lod).get_node(key.minimum)
    }

    /// Mutably borrows the `ChunkNode` for `key`.
    #[inline]
    pub fn get_mut_node(
        &mut self,
        key: ChunkKey<N>,
        drop_chunk: bool,
    ) -> Option<&mut ChunkNode<Usr>> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage_mut(key.lod)
            .get_mut_node(key.minimum, drop_chunk)
    }

    /// Borrows the `NodeState` for the node at `key`. The returned `bool` is `true` iff this node has data.
    #[inline]
    pub fn get_node_state(&self, key: ChunkKey<N>) -> Option<(&NodeState, bool)> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage(key.lod).get_node_state(key.minimum)
    }

    /// Mutably borrows the `NodeState` for the node at `key`. The returned `bool` is `true` iff this node has data.
    #[inline]
    pub fn get_mut_node_state(&mut self, key: ChunkKey<N>) -> Option<(&mut NodeState, bool)> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage_mut(key.lod)
            .get_mut_node_state(key.minimum)
    }

    /// Borrow the chunk at `key`.
    #[inline]
    pub fn get_chunk(&self, key: ChunkKey<N>) -> Option<&Usr> {
        self.get_node(key).and_then(|ch| ch.user_chunk.as_ref())
    }

    /// Mutably borrow the chunk at `key`.
    #[inline]
    pub fn get_mut_chunk(&mut self, key: ChunkKey<N>) -> Option<&mut Usr> {
        self.get_mut_node(key, false)
            .and_then(|c| c.user_chunk.as_mut())
    }

    fn write_node_dangling(&mut self, key: ChunkKey<N>, node: ChunkNode<Usr>) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage_mut(key.lod).insert_node(key.minimum, node);
    }

    fn pop_node_dangling(&mut self, key: ChunkKey<N>, drop_chunk: bool) -> Option<ChunkNode<Usr>> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage_mut(key.lod)
            .pop_node(key.minimum, drop_chunk)
    }

    fn pop_raw_node_dangling(
        &mut self,
        key: ChunkKey<N>,
    ) -> Option<ChunkNode<Either<Store::Chunk, Store::ColdChunk>>> {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        self.lod_storage_mut(key.lod).pop_raw_node(key.minimum)
    }
}

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: ChunkStorage<N>,
{
    /// Call `visitor` on all children keys of `parent_key`.
    #[inline]
    pub fn visit_child_keys(&self, parent_key: ChunkKey<N>, visitor: impl FnMut(ChunkKey<N>, u8)) {
        if let Some((state, _)) = self.get_node_state(parent_key) {
            self.visit_child_keys_of_node(parent_key, state, visitor);
        }
    }

    /// Call `visitor` on all children keys of `parent_key`, reusing the child bitmask on `state` to avoid a hash map lookup.
    #[inline]
    pub fn visit_child_keys_of_node(
        &self,
        parent_key: ChunkKey<N>,
        state: &NodeState,
        mut visitor: impl FnMut(ChunkKey<N>, u8),
    ) {
        for child_i in 0..PointN::NUM_CORNERS {
            if state.children.bit_is_set(child_i) {
                let child_key = self.indexer.child_chunk_key(parent_key, child_i);
                visitor(child_key, child_i);
            }
        }
    }

    /// Call `visitor` on all chunk keys in the subtree with root at `key`.
    ///
    /// A subtree will be pruned from the traversal iff `visitor` returns `false`.
    pub fn visit_tree_keys(&self, key: ChunkKey<N>, mut visitor: impl FnMut(ChunkKey<N>) -> bool) {
        self.visit_tree_keys_recursive(key, &mut visitor);
    }

    fn visit_tree_keys_recursive(
        &self,
        key: ChunkKey<N>,
        visitor: &mut impl FnMut(ChunkKey<N>) -> bool,
    ) {
        let keep_going = visitor(key);
        if keep_going && key.lod > 0 {
            self.visit_child_keys(key, |child_key, _| {
                self.visit_tree_keys_recursive(child_key, visitor);
            });
        }
    }
}

impl<N, T, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: ChunkStorage<N> + for<'r> IterChunkKeys<'r, N>,
    Bldr: ChunkTreeBuilder<N, T>,
{
    /// Call `visitor` on all root chunk keys.
    #[inline]
    pub fn visit_root_keys(&self, mut visitor: impl FnMut(ChunkKey<N>)) {
        let root_lod = self.root_lod();
        for &root in self.lod_storage(root_lod).chunk_keys() {
            visitor(ChunkKey::new(root_lod, root));
        }
    }

    /// Call `visitor` on all keys in the entire tree. This happens in depth-first order.
    ///
    /// A subtree will be pruned from the traversal iff `visitor` returns `false`.
    pub fn visit_all_keys(&self, mut visitor: impl FnMut(ChunkKey<N>) -> bool) {
        self.visit_root_keys(|root_key| {
            self.visit_tree_keys(root_key, &mut visitor);
        })
    }

    /// Returns `true` iff any chunk overlapping `extent` is loading.
    ///
    /// `extent` should be given in voxel coordinates of `lod`. This should be used before editing `extent` to ensure loads
    /// are not interrupted, assuming that edits would like to read-modify-write.
    pub fn extent_is_loading(&self, lod: u8, extent: ExtentN<N>) -> bool {
        let mut loading = false;
        self.visit_root_keys(|root_key| {
            loading |= self.extent_is_loading_recursive(root_key, lod, extent);
        });
        loading
    }

    // Precondition: the node at key exists.
    fn extent_is_loading_recursive(&self, key: ChunkKey<N>, lod: u8, extent: ExtentN<N>) -> bool {
        let lod_extent = self.indexer.chunk_extent_at_lower_lod(key, lod);
        if lod_extent.intersection(&extent).is_empty() {
            // No chunks at lod will overlap extent.
            return false;
        }

        let (node_state, _) = self.get_node_state(key).unwrap();

        if key.lod == lod {
            node_state.is_loading()
        } else {
            for corner_index in 0..PointN::NUM_CORNERS {
                if node_state.descendant_needs_loading.bit_is_set(corner_index) {
                    if node_state.has_child(corner_index) {
                        // We know there are some loading chunks in this tree, but we need to continue traversal to see if they
                        // overlap extent at lod.
                        let child_key = self.indexer.child_chunk_key(key, corner_index);
                        if self.extent_is_loading_recursive(child_key, lod, extent) {
                            return true;
                        }
                    } else {
                        // This entire tree is loading, and it overlaps with extent, so we know at least one loading chunk
                        // overlaps.
                        return true;
                    }
                }
            }
            false
        }
    }
}

impl<N, T, Usr, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Store: ChunkStorage<N, Chunk = Usr>,
{
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
}

impl<N, T, Usr, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
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
}

impl<N, T, Usr, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
    /// Returns `true` iff any chunks in the Moore-neighborhood of `key` are loading.
    pub fn chunk_neighborhood_is_loading(&self, key: ChunkKey<N>) -> bool {
        for offset in PointN::moore_offsets().into_iter() {
            let neighbor_min = key.minimum + offset * self.chunk_shape();
            let neighbor_key = ChunkKey::new(key.lod, neighbor_min);
            if self.node_is_loading(neighbor_key) {
                return true;
            }
        }

        false
    }

    /// Returns `true` iff the node at `key` is currently loading. This can be true even if the node does not exist, in which
    /// case the nearest ancestor node would know if the node is loading (by virtue of being in an empty subtree that is
    /// loading).
    pub fn node_is_loading(&self, key: ChunkKey<N>) -> bool {
        if let Some((state, _)) = self.get_node_state(key) {
            if state.is_loading() {
                return true;
            }
        } else if self.missing_node_is_loading(key) {
            return true;
        }

        false
    }

    /// Iff there is not already a node for `key`, then the entire subtree at `key` will be marked for loading. This means that
    /// a traversal from the ancestor root of `key` will be able to discover all nodes that need to be loaded by following the
    /// `descendant_needs_loading` bits. For an example of this, see [`ChunkTree::clipmap_loading_slots`].
    ///
    /// A node will be considered "loaded" once it's chunk data is mutated in any way, including "no-op" deletions (which is how
    /// we denote that a chunk has no data, even after trying to load it). Once a node is loaded, none of its descendants will
    /// be discoverable for loading until this method is called again. This means nodes should be loaded from the bottom up,
    /// starting with LOD0.
    pub fn mark_tree_for_loading(&mut self, mut key: ChunkKey<N>) {
        let mut already_exists = true;
        self.lod_storage_mut(key.lod)
            .get_mut_node_state_or_insert_with(key.minimum, || {
                already_exists = false;
                ChunkNode::new_loading(key.lod)
            });

        if already_exists {
            return;
        }

        while key.lod < self.root_lod() {
            let parent = self.indexer.parent_chunk_key(key);
            let corner_index = self.indexer.corner_index(key.minimum);
            let (state, _) = self
                .lod_storage_mut(parent.lod)
                .get_mut_node_state_or_insert_with(parent.minimum, ChunkNode::new_empty);
            state.children.set_bit(corner_index);
            state.descendant_needs_loading.set_bit(corner_index);
            key = parent;
        }
    }

    /// Mutably borrow the node at `key`. If the node doesn't exist, `create_chunk` is called to insert one.
    #[inline]
    pub fn get_mut_node_or_insert_chunk_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Usr,
    ) -> &mut ChunkNode<Usr> {
        let (node, _) = self.get_node_for_write(key, false);
        node.user_chunk.get_or_insert_with(create_chunk);
        node
    }

    #[inline]
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Usr,
    ) -> &mut Usr {
        self.get_mut_node_or_insert_chunk_with(key, create_chunk)
            .user_chunk
            .as_mut()
            .unwrap()
    }

    /// Mutably borrow the node at `key`. If the chunk doesn't exist, a new chunk is created with the ambient value.
    #[inline]
    pub fn get_mut_node_or_insert_ambient(&mut self, key: ChunkKey<N>) -> &mut ChunkNode<Usr> {
        let root_lod = self.root_lod();
        let Self {
            indexer,
            builder,
            storages,
            ..
        } = self;
        let (node, _) = Self::_get_node_for_write(storages, indexer, root_lod, key, false);

        node.user_chunk.get_or_insert_with(|| {
            builder.new_ambient(indexer.extent_for_chunk_with_min(key.minimum))
        });

        node
    }

    #[inline]
    pub fn get_mut_chunk_or_insert_ambient(&mut self, key: ChunkKey<N>) -> &mut Usr {
        self.get_mut_node_or_insert_ambient(key)
            .user_chunk
            .as_mut()
            .unwrap()
    }

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value.
    ///
    /// The node's state and a `bool` indicating whether any old data was overwritten are returned for convenience.
    #[inline]
    pub fn write_chunk(&mut self, key: ChunkKey<N>, chunk: Usr) -> (&mut ChunkNode<Usr>, bool) {
        let (node, had_chunk) = self.get_node_for_write(key, true);
        node.user_chunk = Some(chunk);
        (node, had_chunk)
    }

    /// Replace the chunk at `key` with `chunk`, returning the old value.
    #[inline]
    pub fn replace_chunk(&mut self, key: ChunkKey<N>, chunk: Usr) -> Option<Usr> {
        let (node, _) = self.get_node_for_write(key, false);
        node.user_chunk.replace(chunk)
    }

    /// Delete the chunk at `key`, dropping the old value.
    #[inline]
    pub fn delete_chunk(&mut self, key: ChunkKey<N>) {
        let (node, rm) = self.remove_chunk_dangling(key, true);
        self.finish_removal_with_node(key, rm, node);
    }

    /// Remove the chunk at `key`. This does not affect descendant or ancestor chunks.
    #[inline]
    pub fn pop_chunk(&mut self, key: ChunkKey<N>) -> Option<Usr> {
        let (mut node, rm) = self.remove_chunk_dangling(key, false);
        let chunk = node.as_mut().and_then(|n| n.user_chunk.take());
        self.finish_removal_with_node(key, rm, node);
        chunk
    }

    /// Remove the chunk at `key` and all descendants. All chunks will be given to the `chunk_rx` callback.
    ///
    /// Raw chunks are given to `chunk_rx` to avoid any decompression that would happen otherwise.
    pub fn drain_tree(
        &mut self,
        key: ChunkKey<N>,
        chunk_rx: impl FnMut(ChunkKey<N>, ChunkNode<Either<Store::Chunk, Store::ColdChunk>>),
    ) {
        let mut finished_loading_tree = false;
        if let Some(raw_node) = self.pop_raw_node_dangling(key) {
            if raw_node.state.is_loading() {
                finished_loading_tree = true;
            }
            self.drain_tree_recursive(key, raw_node, chunk_rx);
        } else {
            finished_loading_tree = self.missing_node_is_loading(key);
        };
        self.finish_removal(
            key,
            RemoveInfo {
                keep_node: false,
                finished_loading_tree,
            },
        );
    }

    fn drain_tree_recursive(
        &mut self,
        key: ChunkKey<N>,
        node: ChunkNode<Either<Store::Chunk, Store::ColdChunk>>,
        mut chunk_rx: impl FnMut(ChunkKey<N>, ChunkNode<Either<Store::Chunk, Store::ColdChunk>>),
    ) {
        for child_i in 0..PointN::NUM_CORNERS {
            if node.state.has_child(child_i) {
                let child_key = self.indexer.child_chunk_key(key, child_i);
                if let Some(child_node) = self.pop_raw_node_dangling(child_key) {
                    self.drain_tree_recursive(child_key, child_node, &mut chunk_rx);
                }
            }
        }
        chunk_rx(key, node);
    }

    fn get_node_for_write(
        &mut self,
        key: ChunkKey<N>,
        drop_chunk: bool,
    ) -> (&mut ChunkNode<Usr>, bool) {
        let root_lod = self.root_lod();
        let Self {
            indexer, storages, ..
        } = self;
        Self::_get_node_for_write(storages, indexer, root_lod, key, drop_chunk)
    }

    /// This assumes the the caller is going to write the chunk, marks the node as loaded, and links it into the tree.
    fn _get_node_for_write<'a>(
        storages: &'a mut [Store],
        indexer: &ChunkIndexer<N>,
        root_lod: u8,
        key: ChunkKey<N>,
        drop_chunk: bool,
    ) -> (&'a mut ChunkNode<Usr>, bool) {
        debug_assert!(indexer.chunk_min_is_valid(key.minimum));

        let (tree_storages, ancestor_storages) = storages.split_at_mut(key.lod as usize + 1);

        let (node, had_chunk) = tree_storages[key.lod as usize].get_mut_node_or_insert_with(
            key.minimum,
            drop_chunk,
            || {
                if Self::_missing_node_is_loading(ancestor_storages, indexer, root_lod, key) {
                    ChunkNode::new_loading(key.lod)
                } else {
                    ChunkNode::new_empty()
                }
            },
        );

        if !had_chunk {
            let was_loading = node.state.mark_loaded();
            let finished_loading_tree = was_loading && !node.state.tree_is_loading();

            if finished_loading_tree {
                let (parent_key, _, _, new_parent) = Self::_mark_tree_as_loaded_on_parent(
                    &mut ancestor_storages[0],
                    indexer,
                    root_lod,
                    key,
                    true,
                );
                if new_parent {
                    Self::_link_to_nearest_ancestor(
                        &mut ancestor_storages[1..],
                        indexer,
                        root_lod,
                        parent_key,
                        ChunkNode::new_loading_parent,
                    );
                }
            } else {
                Self::_link_to_nearest_ancestor(
                    ancestor_storages,
                    indexer,
                    root_lod,
                    key,
                    ChunkNode::new_empty,
                );
            }
        }

        (node, had_chunk)
    }

    /// This must be used for any method that removes or attempts to remove a chunk from the tree. The `RemoveInfo` must be used
    /// to clean up after this operation with `finish_removal` or `finish_removal_with_node`.
    fn remove_chunk_dangling(
        &mut self,
        key: ChunkKey<N>,
        drop_chunk: bool,
    ) -> (Option<ChunkNode<Usr>>, RemoveInfo) {
        debug_assert!(self.indexer.chunk_min_is_valid(key.minimum));

        let root_lod = self.root_lod();
        let Self {
            indexer, storages, ..
        } = self;
        let (tree_storages, ancestor_storages) = storages.split_at_mut(key.lod as usize + 1);

        let (node, keep_node, finished_loading_tree) =
            if let Some(node) = tree_storages[key.lod as usize].pop_node(key.minimum, drop_chunk) {
                let keep_node = node.state.descendants_have_data();
                let was_loading = node.state.mark_loaded();
                let finished_loading_tree = was_loading && !node.state.tree_is_loading();
                (Some(node), keep_node, finished_loading_tree)
            } else if Self::_missing_node_is_loading(ancestor_storages, indexer, root_lod, key) {
                let finished_loading = key.lod == 0;
                let node = (!finished_loading).then(ChunkNode::new_loading_parent);
                (node, !finished_loading, finished_loading)
            } else {
                (None, false, false)
            };

        (
            node,
            RemoveInfo {
                keep_node,
                finished_loading_tree,
            },
        )
    }

    fn finish_removal_with_node(
        &mut self,
        key: ChunkKey<N>,
        info: RemoveInfo,
        node: Option<ChunkNode<Usr>>,
    ) {
        if info.keep_node {
            if let Some(node) = node {
                self.lod_storage_mut(key.lod).insert_node(key.minimum, node);
            }
        }
        self.finish_removal(key, info);
    }

    fn finish_removal(&mut self, mut key: ChunkKey<N>, info: RemoveInfo) {
        let root_lod = self.root_lod();
        let Self {
            indexer, storages, ..
        } = self;
        let (_, ancestor_storages) = storages.split_at_mut(key.lod as usize + 1);

        // Recursively remove any nodes whose trees don't have data. Otherwise link the nodes that do.
        let mut keep_node = info.keep_node;
        let mut mark_loaded_on_parent = info.finished_loading_tree;
        let mut storage_index = 0;
        while key.lod < root_lod {
            let parent_key = indexer.parent_chunk_key(key);
            let corner_index = indexer.corner_index(key.minimum);

            // Make changes to the parent node.
            let mut new_parent = false;
            let parent_state = if keep_node || mark_loaded_on_parent {
                let (parent_state, parent_has_chunk) = ancestor_storages[storage_index]
                    .get_mut_node_state_or_insert_with(parent_key.minimum, || {
                        new_parent = true;
                        ChunkNode::new_loading_parent()
                    });

                if keep_node {
                    parent_state.children.set_bit(corner_index);
                } else {
                    parent_state.children.unset_bit(corner_index);
                }

                if mark_loaded_on_parent {
                    parent_state
                        .descendant_needs_loading
                        .unset_bit(corner_index);
                }

                Some((parent_state, parent_has_chunk))
            } else {
                // We don't need to create a parent if it doesn't exist, since we're not keeping the child or marking it as
                // loaded.
                ancestor_storages[storage_index]
                    .get_mut_node_state(parent_key.minimum)
                    .map(|(parent_state, parent_has_chunk)| {
                        parent_state.children.unset_bit(corner_index);
                        (parent_state, parent_has_chunk)
                    })
            };

            if let Some((parent_state, parent_has_chunk)) = parent_state {
                keep_node = parent_has_chunk || parent_state.descendants_have_data();
                mark_loaded_on_parent = mark_loaded_on_parent && !parent_state.tree_is_loading();

                if !keep_node {
                    ancestor_storages[storage_index].pop_raw_node(parent_key.minimum);
                } else if !new_parent && !mark_loaded_on_parent {
                    // This parent was already linked, it's not being removed, and we don't need to mark it as loaded.
                    return;
                }
            } else {
                // No parent, so we don't need to make any more changes.
                return;
            }

            key = parent_key;
            storage_index += 1;
        }
    }

    fn missing_node_is_loading(&self, key: ChunkKey<N>) -> bool {
        Self::_missing_node_is_loading(
            &self.storages[key.lod as usize + 1..],
            &self.indexer,
            self.root_lod(),
            key,
        )
    }

    fn _missing_node_is_loading(
        ancestor_storages: &[Store],
        indexer: &ChunkIndexer<N>,
        root_lod: u8,
        key: ChunkKey<N>,
    ) -> bool {
        if let Some((ancestor_state, path_corner_index)) =
            Self::_find_nearest_ancestor_node(ancestor_storages, indexer, root_lod, key)
        {
            ancestor_state
                .descendant_needs_loading
                .bit_is_set(path_corner_index)
        } else {
            false
        }
    }

    fn _find_nearest_ancestor_node<'a>(
        ancestor_storages: &'a [Store],
        indexer: &ChunkIndexer<N>,
        root_lod: u8,
        mut key: ChunkKey<N>,
    ) -> Option<(&'a NodeState, u8)> {
        let mut storage_index = 0;
        while key.lod < root_lod {
            let parent_key = indexer.parent_chunk_key(key);
            if let Some((parent_state, _)) =
                ancestor_storages[storage_index].get_node_state(parent_key.minimum)
            {
                let corner_index = indexer.corner_index(key.minimum);
                return Some((parent_state, corner_index));
            }
            key = parent_key;
            storage_index += 1;
        }

        None
    }

    fn _link_to_nearest_ancestor(
        ancestor_storages: &mut [Store],
        indexer: &ChunkIndexer<N>,
        root_lod: u8,
        mut key: ChunkKey<N>,
        make_link_node: impl Fn() -> ChunkNode<Usr>,
    ) {
        let mut storage_index = 0;
        while key.lod < root_lod {
            let parent = indexer.parent_chunk_key(key);
            let mut parent_already_exists = true;
            let (parent_state, _) = ancestor_storages[storage_index]
                .get_mut_node_state_or_insert_with(parent.minimum, || {
                    parent_already_exists = false;
                    make_link_node()
                });
            parent_state
                .children
                .set_bit(indexer.corner_index(key.minimum));
            if parent_already_exists {
                return;
            }
            key = parent;
            storage_index += 1;
        }
    }

    fn _mark_tree_as_loaded_on_parent<'a>(
        parent_lod_storage: &'a mut Store,
        indexer: &ChunkIndexer<N>,
        root_lod: u8,
        child_key: ChunkKey<N>,
        child_node_exists: bool,
    ) -> (ChunkKey<N>, &'a mut NodeState, bool, bool) {
        debug_assert!(child_key.lod < root_lod);

        let parent = indexer.parent_chunk_key(child_key);
        let corner_index = indexer.corner_index(child_key.minimum);
        let mut new_node = false;
        let (state, has_chunk) =
            parent_lod_storage.get_mut_node_state_or_insert_with(parent.minimum, || {
                new_node = true;
                ChunkNode::new_loading_parent()
            });
        state.descendant_needs_loading.unset_bit(corner_index);
        if child_node_exists {
            state.children.set_bit(corner_index);
        } else {
            state.children.unset_bit(corner_index);
        }
        (parent, state, has_chunk, new_node)
    }
}

struct RemoveInfo {
    keep_node: bool,
    finished_loading_tree: bool,
}

/// A node in the `ChunkTree`.
#[derive(Clone, Deserialize, Serialize)]
pub struct ChunkNode<U> {
    /// Parent chunks are `None` until written or downsampled into. This means that users can opt-in to storing downsampled
    /// chunks, which requires more memory.
    pub user_chunk: Option<U>,
    /// Chunk-related state. See [`NodeState`].
    pub state: NodeState,
}

impl<U> ChunkNode<U> {
    #[inline]
    pub fn as_ref(&self) -> ChunkNode<&U> {
        ChunkNode::new(self.user_chunk.as_ref(), self.state.clone())
    }

    #[inline]
    pub fn map<T>(self, f: impl Fn(U) -> T) -> ChunkNode<T> {
        ChunkNode::new(self.user_chunk.map(f), self.state)
    }

    #[inline]
    pub fn new(user_chunk: Option<U>, state: NodeState) -> Self {
        Self { user_chunk, state }
    }

    #[inline]
    pub fn new_empty() -> Self {
        Self::new(None, NodeState::default())
    }

    #[inline]
    pub fn new_loading(lod: u8) -> Self {
        if lod == 0 {
            Self::new_loading_leaf()
        } else {
            Self::new_loading_parent()
        }
    }

    #[inline]
    pub fn new_loading_leaf() -> Self {
        let state = NodeState::default();
        state.state_bits.set_bit(StateBit::Loading as u8);
        Self::new_without_data(state)
    }

    #[inline]
    pub fn new_loading_parent() -> Self {
        let mut state = NodeState::default();
        state.descendant_needs_loading.set_all();
        state.state_bits.set_bit(StateBit::Loading as u8);
        Self::new_without_data(state)
    }

    #[inline]
    pub fn new_without_data(state: NodeState) -> Self {
        Self::new(None, state)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct NodeState {
    /// A bitmask tracking which child nodes exist.
    children: Bitset8,
    /// A bitmask to help with searching for nodes that need to be loaded.
    ///
    /// A node may only be a downsample target if all of its children are loaded. Leaf nodes may only be missing data if these
    /// bits are not all set.
    pub descendant_needs_loading: Bitset8,
    /// A bitmask tracking other external state, like if the chunk is being rendered.
    ///
    /// Check [`StateBit`] to see which bits are being used internally.
    pub state_bits: AtomicBitset8,
}

impl NodeState {
    /// Returns `true` iff the child slot at `corner_index` has a node in the tree.
    #[inline]
    pub fn has_child(&self, corner_index: u8) -> bool {
        self.children.bit_is_set(corner_index)
    }

    /// Returns `true` iff any child slots have a node in the tree.
    #[inline]
    pub fn has_any_children(&self) -> bool {
        self.children.any()
    }

    /// Returns `true` iff any children exist or are loading.
    #[inline]
    pub fn descendants_have_data(&self) -> bool {
        self.children.any() || self.descendant_needs_loading.any()
    }

    #[inline]
    pub fn is_loading(&self) -> bool {
        self.state_bits.bit_is_set(StateBit::Loading as u8)
    }

    #[inline]
    pub fn tree_is_loading(&self) -> bool {
        self.is_loading() || self.descendant_needs_loading.any()
    }

    #[inline]
    pub fn mark_loaded(&self) -> bool {
        self.state_bits.fetch_and_unset_bit(StateBit::Loading as u8)
    }

    #[inline]
    pub fn is_rendering(&self) -> bool {
        self.state_bits.bit_is_set(StateBit::Render as u8)
    }
}

#[repr(u8)]
pub enum StateBit {
    /// This bit is set if the chunk is currently loading.
    Loading = 0,
    /// This bit is set if the chunk is currently being rendered.
    Render = 1,
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
    const MAP_CONFIG: ChunkTreeConfig<[i32; 3], i32> = ChunkTreeConfig {
        chunk_shape: CHUNK_SHAPE,
        ambient_value: 0,
        root_lod: 2,
    };
    const MULTICHAN_MAP_CONFIG: ChunkTreeConfig<[i32; 3], (i32, u8)> = ChunkTreeConfig {
        chunk_shape: CHUNK_SHAPE,
        ambient_value: (0, b'a'),
        root_lod: 0,
    };

    #[test]
    fn write_and_read_points() {
        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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
        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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

        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

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
        let builder = ChunkTreeBuilder3x2::new(MULTICHAN_MAP_CONFIG);
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

        map.lod_view_mut(0).fill_extent(&extent, (1, b'b'));
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn multichannel_compressed_accessors() {
        use crate::prelude::{FastCompressibleChunkStorageNx2, Lz4};

        let builder = ChunkTreeBuilder3x2::new(MULTICHAN_MAP_CONFIG);
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

    #[test]
    fn hash_map_chunk_tree_can_serde() {
        let builder = ChunkTreeBuilder3x2::new(MULTICHAN_MAP_CONFIG);
        let map = builder.build_with_hash_map_storage();
        can_serde(map);
    }

    fn can_serde<'a, T>(_x: T)
    where
        T: Deserialize<'a> + Serialize,
    {
    }

    #[test]
    fn load_and_edit_bottom_up() {
        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

        let load_lod = 1;
        let load_key = ChunkKey::new(load_lod, PointN::ZERO);
        map.mark_tree_for_loading(load_key);

        let load_key_extent = map.indexer.extent_for_chunk_with_min(load_key.minimum);
        let partial_overlap = load_key_extent + map.chunk_shape() / 2;

        assert!(map.extent_is_loading(load_lod, partial_overlap));

        let loading_extent_lod0 = map.indexer.chunk_extent_at_lower_lod(load_key, 0);
        let mut lod0 = map.lod_view_mut(0);
        lod0.fill_extent(&loading_extent_lod0, 1);

        // No longer loading at LOD0.
        assert!(!map.extent_is_loading(0, loading_extent_lod0));
        // Still loading at LOD1.
        assert!(map.extent_is_loading(load_lod, load_key_extent));

        let other_key = ChunkKey::new(load_key.lod, load_key.minimum + map.chunk_shape());
        map.write_chunk(other_key, Array3x1::fill(load_key_extent, 1));

        // Still loading at LOD1.
        assert!(map.extent_is_loading(load_lod, load_key_extent));

        map.write_chunk(load_key, Array3x1::fill(load_key_extent, 1));

        // Done loading at LOD1.
        assert!(!map.extent_is_loading(load_lod, load_key_extent));
    }

    #[test]
    fn load_and_delete_top_down() {
        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

        let load_lod = 1;
        let load_key = ChunkKey::new(load_lod, PointN::ZERO);
        map.mark_tree_for_loading(load_key);

        let load_key_extent = map.indexer.extent_for_chunk_with_min(load_key.minimum);
        let partial_overlap = load_key_extent + map.chunk_shape() / 2;

        assert!(map.extent_is_loading(load_lod, partial_overlap));

        // Load the key as empty.
        map.delete_chunk(load_key);

        // Done loading at LOD1.
        assert!(!map.extent_is_loading(load_lod, load_key_extent));

        // Still loading at LOD0.
        let loading_extent_lod0 = map.indexer.chunk_extent_at_lower_lod(load_key, 0);
        assert!(map.extent_is_loading(0, loading_extent_lod0));

        let mut lod0 = map.lod_view_mut(0);
        lod0.fill_extent(&loading_extent_lod0, 1);

        // No longer loading at LOD0.
        assert!(!map.extent_is_loading(0, loading_extent_lod0));
    }

    #[test]
    fn delete_collapses_tree() {
        let mut map = ChunkTreeBuilder3x1::new(MAP_CONFIG).build_with_hash_map_storage();

        let root_key = ChunkKey::new(2, PointN::ZERO);
        let lod0_extent = map.indexer.chunk_extent_at_lower_lod(root_key, 0);
        map.lod_view_mut(0).fill_extent(&lod0_extent, 1);

        let mut delete_keys = Vec::new();
        map.visit_all_keys(|key| {
            if key.lod == 0 {
                delete_keys.push(key);
            }
            true
        });

        for key in delete_keys.into_iter() {
            map.delete_chunk(key);
        }

        map.visit_root_keys(|_| panic!("expected no roots"));
    }
}
