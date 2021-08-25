use super::child_mask_has_child;
use crate::dev_prelude::*;

use building_blocks_core::{prelude::*, Sphere};

use std::collections::HashSet;

impl<Ni, Nf, T, Usr, Bldr, Store> ChunkTree<Ni, T, Bldr, Store>
where
    PointN<Ni>: std::hash::Hash + IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
    Bldr: ChunkTreeBuilder<Ni, T, Chunk = Usr>,
    Store: ChunkStorage<Ni, Chunk = Usr> + for<'r> IterChunkKeys<'r, Ni>,
{
    /// Traverses from all roots to find the currently "active" chunks. Active chunks are passed to the `active_rx` callback.
    ///
    /// By "active," we mean that, given the current location of `clip_sphere`:
    ///
    ///   - the chunk is bounded by the clip sphere determined by `clip_sphere.radius`
    ///   - the chunk has the desired LOD for rendering
    ///
    /// More specifically,
    ///
    ///   - let `D` be the Euclidean distance from `clip_sphere.center` to the center of the chunk (in LOD0 space)
    ///   - let `B` be the radius of the chunk's bounding sphere (in LOD0 space)
    ///   - let `S` be the shape of the chunk (in LOD0 space)
    ///
    /// The chunk *can* be active iff
    ///
    /// ```text
    ///     D + B < clip_sphere.radius && (D / S) > detail
    /// ```
    ///
    /// where `detail` is a nonnegative constant parameter supplied by you. Along a given path from a root chunk to a leaf, the
    /// least detailed chunk that *can* be active is used.
    pub fn clipmap_active_chunks(
        &self,
        detail: f32,
        clip_sphere: Sphere<Nf>,
        mut active_rx: impl FnMut(ChunkKey<Ni>),
    ) {
        assert!(clip_sphere.radius > 0.0);

        let root_lod = self.root_lod();
        let root_storage = self.lod_storage(root_lod);
        for &chunk_min in root_storage.chunk_keys() {
            self.clipmap_active_chunks_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                clip_sphere,
                &mut active_rx,
            );
        }
    }

    fn clipmap_active_chunks_recursive(
        &self,
        node_key: ChunkKey<Ni>,
        detail: f32,
        clip_sphere: Sphere<Nf>,
        active_rx: &mut impl FnMut(ChunkKey<Ni>),
    ) {
        let node_lod0_extent = self.indexer.chunk_extent_at_lower_lod(node_key, 0);
        let node_lod0_center = node_lod0_extent.minimum + (node_lod0_extent.shape >> 1);
        let node_lod0_radius = (node_lod0_extent.shape.max_component() >> 1) as f32 * 3f32.sqrt();

        // Calculate the Euclidean distance from the focus the center of the chunk.
        let dist = clip_sphere
            .center
            .l2_distance_squared(PointN::<Nf>::from(node_lod0_center))
            .sqrt();

        let node_intersects_clip_sphere = dist - node_lod0_radius < clip_sphere.radius;

        // Don't consider any chunk that doesn't intersect the clip sphere.
        if !node_intersects_clip_sphere {
            return;
        }

        let node_bounded_by_clip_sphere = dist + node_lod0_radius < clip_sphere.radius;

        // Normalize distance by chunk shape.
        let is_active = node_key.lod == 0 || (dist / node_lod0_radius) > detail;

        if is_active {
            if node_bounded_by_clip_sphere {
                // This node is active!
                active_rx(node_key);
            }
        } else {
            // Need to split, continue by recursing on the children.
            let child_mask = self.get_child_mask(node_key).unwrap();
            for child_i in 0..PointN::NUM_CORNERS {
                if child_mask_has_child(child_mask, child_i) {
                    self.clipmap_active_chunks_recursive(
                        self.indexer.child_chunk_key(node_key, child_i),
                        detail,
                        clip_sphere,
                        active_rx,
                    );
                }
            }
        }
    }

    /// Detects new chunk slots that entered `new_clip_sphere` after it moved from `old_clip_sphere`.
    ///
    /// `min_lod` will be used to stop the search earlier than LOD0 when an *active* ancestor has already entered. This
    /// significantly improves performance for the branching and bounding algorithm. When configured, the user must assume that
    /// if an enter event is received at any LOD less than or equal to `min_lod`, it also applies to all descendants of the
    /// given chunk key.
    pub fn clipmap_new_chunks(
        &self,
        detail: f32,
        min_lod: u8,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        mut rx: impl FnMut(NewChunkSlot<Ni>),
    ) {
        assert!(old_clip_sphere.radius > 0.0);
        assert!(new_clip_sphere.radius > 0.0);

        let root_lod = self.root_lod();

        let old_lod0_clip_extent = old_clip_sphere.aabb().containing_integer_extent();
        let new_lod0_clip_extent = new_clip_sphere.aabb().containing_integer_extent();
        let old_root_clip_extent = self
            .indexer
            .covering_ancestor_extent(old_lod0_clip_extent, root_lod as i32);
        let new_root_clip_extent = self
            .indexer
            .covering_ancestor_extent(new_lod0_clip_extent, root_lod as i32);

        // Union the root nodes covering both clip spheres.
        let root_nodes: HashSet<PointN<Ni>> = self
            .indexer
            .chunk_mins_for_extent(&old_root_clip_extent)
            .chain(self.indexer.chunk_mins_for_extent(&new_root_clip_extent))
            .collect();

        for chunk_min in root_nodes.into_iter() {
            self.clipmap_new_chunks_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                min_lod,
                old_clip_sphere,
                new_clip_sphere,
                false,
                false,
                &mut rx,
            );
        }
    }

    fn clipmap_new_chunks_recursive(
        &self,
        node_key: ChunkKey<Ni>, // May not exist in the ChunkTree!
        detail: f32,
        min_lod: u8,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        ancestor_was_active: bool,
        ancestor_is_active: bool,
        rx: &mut impl FnMut(NewChunkSlot<Ni>),
    ) {
        if ancestor_was_active && ancestor_is_active && node_key.lod < min_lod {
            // We can't have more active enters in this tree, and user said it's OK to skip the inactive ones at this LOD.
            return;
        }

        let info = ChunkSphereInfo::new(
            &self.indexer,
            old_clip_sphere.center,
            new_clip_sphere.center,
            node_key,
        );

        let node_intersects_old_clip_sphere =
            info.dist_to_old_clip_sphere - info.lod0_radius < old_clip_sphere.radius;
        let node_intersects_new_clip_sphere =
            info.dist_to_new_clip_sphere - info.lod0_radius < new_clip_sphere.radius;

        if !node_intersects_old_clip_sphere && !node_intersects_new_clip_sphere {
            // There are no events for this node or any of its descendants.
            return;
        }

        let node_bounded_by_old_clip_sphere =
            info.dist_to_old_clip_sphere + info.lod0_radius < old_clip_sphere.radius;
        let node_bounded_by_new_clip_sphere =
            info.dist_to_new_clip_sphere + info.lod0_radius < new_clip_sphere.radius;

        if node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere {
            // This node is stably bounded, so no enter events are possible.
            return;
        }

        let was_active =
            !ancestor_was_active && (node_key.lod == 0 || info.old_normalized_distance() > detail);
        let is_active =
            !ancestor_is_active && (node_key.lod == 0 || info.new_normalized_distance() > detail);

        // Recurse.
        if node_key.lod > 0 {
            for child_i in 0..PointN::NUM_CORNERS {
                self.clipmap_new_chunks_recursive(
                    self.indexer.child_chunk_key(node_key, child_i),
                    detail,
                    min_lod,
                    old_clip_sphere,
                    new_clip_sphere,
                    ancestor_was_active || was_active,
                    ancestor_is_active || is_active,
                    rx,
                );
            }
        }

        // We check for enter events after recursing because we need to guarantee all descendant enter events come first.
        if !node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere {
            rx(NewChunkSlot {
                key: node_key,
                is_active,
            });
        }
    }

    pub fn lod_changes(
        &self,
        detail: f32,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        mut rx: impl FnMut(LodChange<Ni>),
    ) {
        assert!(old_clip_sphere.radius > 0.0);
        assert!(new_clip_sphere.radius > 0.0);

        let root_lod = self.root_lod();
        if root_lod == 0 {
            return;
        }

        let root_storage = self.lod_storage(root_lod);
        for &chunk_min in root_storage.chunk_keys() {
            self.lod_changes_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                old_clip_sphere,
                new_clip_sphere,
                &mut rx,
            );
        }
    }

    fn lod_changes_recursive(
        &self,
        node_key: ChunkKey<Ni>, // Precondition: not LOD0.
        detail: f32,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        rx: &mut impl FnMut(LodChange<Ni>),
    ) {
        let info = ChunkSphereInfo::new(
            &self.indexer,
            old_clip_sphere.center,
            new_clip_sphere.center,
            node_key,
        );

        let node_intersects_old_clip_sphere =
            info.dist_to_old_clip_sphere - info.lod0_radius < old_clip_sphere.radius;
        let node_intersects_new_clip_sphere =
            info.dist_to_new_clip_sphere - info.lod0_radius < new_clip_sphere.radius;

        if !node_intersects_old_clip_sphere && !node_intersects_new_clip_sphere {
            return;
        }

        // We'll detect split/merge even for chunks that aren't totally bounded by the clip sphere. Some of their children might
        // be bounded, and we want to know if they become active/inactive.

        let was_active = info.old_normalized_distance() > detail;
        let is_active = info.new_normalized_distance() > detail;

        match (was_active, is_active) {
            // Old and new frames agree this chunk is not active.
            (false, false) => {
                // Keep looking for any active descendants.
                if node_key.lod > 1 {
                    self.for_each_child(node_key, |child_key| {
                        self.lod_changes_recursive(
                            child_key,
                            detail,
                            old_clip_sphere,
                            new_clip_sphere,
                            rx,
                        );
                    });
                }
            }
            (false, true) => {
                // This node just became active, and none if its ancestors were active, so it must have active descendants.
                // Merge those active descendants into this node.
                let mut old_chunks = Vec::with_capacity(8);
                self.for_each_child(node_key, |child_key| {
                    self.clipmap_active_chunks_recursive(
                        child_key,
                        detail,
                        old_clip_sphere,
                        &mut |active_chunk| {
                            // Only add the chunk if it didn't enter this
                            old_chunks.push(active_chunk);
                        },
                    );
                });

                let new_chunk_is_bounded =
                    info.dist_to_new_clip_sphere + info.lod0_radius < new_clip_sphere.radius;

                rx(LodChange::Merge(MergeChunks {
                    old_chunks,
                    new_chunk: node_key,
                    new_chunk_is_bounded,
                }));
            }
            (true, false) => {
                // This node just became inactive, and none of its ancestors were active, so it must have active descendants.
                // Split this node into the children.
                let mut new_chunks = Vec::with_capacity(8);
                self.for_each_child(node_key, |child_key| {
                    self.clipmap_active_chunks_recursive(
                        child_key,
                        detail,
                        new_clip_sphere,
                        &mut |active_chunk| new_chunks.push(active_chunk),
                    );
                });

                let old_chunk_was_bounded =
                    info.dist_to_old_clip_sphere + info.lod0_radius < old_clip_sphere.radius;

                rx(LodChange::Split(SplitChunk {
                    old_chunk: node_key,
                    old_chunk_was_bounded,
                    new_chunks,
                }));
            }
            // Old and new agree this node is active. No need to merge or split. None of the descendants can merge or split
            // either.
            (true, true) => (),
        }
    }
}

/// Some common intermediate calculations about a chunk.
#[derive(Debug)]
struct ChunkSphereInfo {
    lod0_radius: f32,
    dist_to_old_clip_sphere: f32,
    dist_to_new_clip_sphere: f32,
}

impl ChunkSphereInfo {
    fn new<Ni, Nf>(
        indexer: &ChunkIndexer<Ni>,
        old_focus: PointN<Nf>,
        new_focus: PointN<Nf>,
        node_key: ChunkKey<Ni>,
    ) -> Self
    where
        PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
        PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
    {
        let node_sphere = chunk_lod0_bounding_sphere(indexer, node_key);

        // Calculate the Euclidean distance from each focus the center of the chunk.
        let dist_to_old_clip_sphere = old_focus.l2_distance_squared(node_sphere.center).sqrt();
        let dist_to_new_clip_sphere = new_focus.l2_distance_squared(node_sphere.center).sqrt();

        Self {
            lod0_radius: node_sphere.radius,
            dist_to_old_clip_sphere,
            dist_to_new_clip_sphere,
        }
    }

    fn old_normalized_distance(&self) -> f32 {
        self.dist_to_old_clip_sphere / self.lod0_radius
    }

    fn new_normalized_distance(&self) -> f32 {
        self.dist_to_new_clip_sphere / self.lod0_radius
    }
}

pub fn chunk_lod0_bounding_sphere<Ni, Nf>(
    indexer: &ChunkIndexer<Ni>,
    chunk_key: ChunkKey<Ni>,
) -> Sphere<Nf>
where
    PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
{
    let node_lod0_extent = indexer.chunk_extent_at_lower_lod(chunk_key, 0);
    let center = PointN::<Nf>::from(node_lod0_extent.minimum + (node_lod0_extent.shape >> 1));

    let shape_max_comp = node_lod0_extent.shape.max_component();
    let radius = (shape_max_comp >> 1) as f32 * 3f32.sqrt();

    Sphere { center, radius }
}

/// A new chunk slot has entered the clip sphere.
pub struct NewChunkSlot<N> {
    pub key: ChunkKey<N>,
    pub is_active: bool,
}

/// A 2-dimensional `NewChunkSlot`.
pub type NewChunkSlot2 = NewChunkSlot<[i32; 2]>;
/// A 3-dimensional `NewChunkSlot`.
pub type NewChunkSlot3 = NewChunkSlot<[i32; 3]>;

/// A chunk's desired sample rate has changed based on proximity to the center of the clip sphere.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodChange<N> {
    /// The desired sample rate for this chunk increased this frame.
    Split(SplitChunk<N>),
    /// The desired sample rate for this chunk decreased this frame.
    Merge(MergeChunks<N>),
}

/// A 2-dimensional `LodChange`.
pub type LodChange2 = LodChange<[i32; 2]>;
/// A 3-dimensional `LodChange`.
pub type LodChange3 = LodChange<[i32; 3]>;

/// Split `old_chunk` into many `new_chunks`. The number of new chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitChunk<N> {
    pub old_chunk: ChunkKey<N>,
    pub old_chunk_was_bounded: bool,
    pub new_chunks: Vec<ChunkKey<N>>,
}

/// Merge many `old_chunks` into `new_chunk`. The number of old chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeChunks<N> {
    pub old_chunks: Vec<ChunkKey<N>>,
    pub new_chunk: ChunkKey<N>,
    pub new_chunk_is_bounded: bool,
}
