use super::child_mask_has_child;
use crate::dev_prelude::*;

use building_blocks_core::prelude::*;

impl<Ni, Nf, T, Usr, Bldr, Store> ChunkTree<Ni, T, Bldr, Store>
where
    PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
    Bldr: ChunkTreeBuilder<Ni, T, Chunk = Usr>,
    Store: ChunkStorage<Ni, Chunk = Usr> + for<'r> IterChunkKeys<'r, Ni>,
{
    /// Traverses from all roots to find the currently "active" chunks. Active chunks are passed to the `active_rx` callback.
    ///
    /// By "active," we mean that, given the current location of `lod0_focus` (e.g. a camera), the chunk has the desired LOD for
    /// rendering. More specifically, let `D` be the Euclidean distance from `lod0_focus` to the center of the chunk (in LOD0
    /// space), and let `S` be the shape of the chunk (in LOD0 space). The chunk can be active if
    ///
    /// ```text
    ///     (D / S) > detail
    /// ```
    ///
    /// where `detail` is a nonnegative constant parameter supplied by you. Along a given path from a root chunk to a leaf, the
    /// least detailed chunk that *can* be active is used.
    ///
    /// `predicate` is used to terminate traversal of a subtree early so that the rejected chunk and any of its descendants
    /// cannot be considered active. This could be useful for e.g. culling chunks outside of a view frustum.
    pub fn clipmap_active_chunks(
        &self,
        detail: f32,
        lod0_focus: PointN<Nf>,
        mut predicate: impl FnMut(ChunkKey<Ni>) -> bool,
        mut active_rx: impl FnMut(ChunkKey<Ni>),
    ) {
        let root_lod = self.root_lod();
        let root_storage = self.lod_storage(root_lod);
        for &chunk_min in root_storage.chunk_keys() {
            self.clipmap_active_chunks_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                lod0_focus,
                &mut predicate,
                &mut active_rx,
            );
        }
    }

    fn clipmap_active_chunks_recursive(
        &self,
        node_key: ChunkKey<Ni>,
        detail: f32,
        lod0_focus: PointN<Nf>,
        predicate: &mut impl FnMut(ChunkKey<Ni>) -> bool,
        active_rx: &mut impl FnMut(ChunkKey<Ni>),
    ) {
        if !predicate(node_key) {
            return;
        }

        if node_key.lod == 0 {
            // This node is active!
            active_rx(node_key);
        }

        let is_active = self.normalized_dist(lod0_focus, node_key) > PointN::fill(detail);
        if is_active {
            // This node is active!
            active_rx(node_key);
        } else {
            let child_mask = self.get_child_mask(node_key).unwrap();
            // Need to split, continue by recursing on the children.
            for child_i in 0..PointN::NUM_CORNERS {
                if child_mask_has_child(child_mask, child_i) {
                    self.clipmap_active_chunks_recursive(
                        self.indexer.child_chunk_key(node_key, child_i),
                        detail,
                        lod0_focus,
                        predicate,
                        active_rx,
                    );
                }
            }
        }
    }

    /// Like `active_clipmap_chunks`, but it only detects changes in the set of active chunks after the focal point moves from
    /// `old_lod0_focus` to `new_lod0_focus`.
    ///
    /// In order to detect new chunk slots to be loaded and old chunk slots to evict, the user must provide a
    /// `lod0_clip_radius`. You can think of this as representing a cube around the focus where chunk slots can enter and exit
    /// the cube as the focus moves.
    pub fn clipmap_events(
        &self,
        detail: f32,
        lod0_clip_radius: f32,
        old_lod0_focus: PointN<Nf>,
        new_lod0_focus: PointN<Nf>,
        mut predicate: impl FnMut(ChunkKey<Ni>) -> bool,
        mut event_rx: impl FnMut(ClipEvent<Ni>),
    ) {
        assert!(lod0_clip_radius > 0.0);
        let lod0_clip_extent = ExtentN::from_min_and_shape(
            PointN::fill(-lod0_clip_radius),
            PointN::fill(2.0 * lod0_clip_radius),
        );
        let old_lod0_clip_extent = (lod0_clip_extent + old_lod0_focus).containing_integer_extent();
        let new_lod0_clip_extent = (lod0_clip_extent + new_lod0_focus).containing_integer_extent();
        // This extent covers both the old and new extents, ensuring we don't miss any relevant events.
        let union_lod0_clip_extent = old_lod0_clip_extent.quasi_union(&new_lod0_clip_extent);

        let root_lod = self.root_lod();
        let root_clip_extent = self
            .indexer
            .covering_ancestor_extent(union_lod0_clip_extent, root_lod as i32);
        for chunk_min in self.indexer.chunk_mins_for_extent(&root_clip_extent) {
            self.clipmap_events_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                old_lod0_focus,
                new_lod0_focus,
                &old_lod0_clip_extent,
                &new_lod0_clip_extent,
                &mut predicate,
                &mut event_rx,
            );
        }
    }

    fn clipmap_events_recursive(
        &self,
        node_key: ChunkKey<Ni>, // May not exist in the ChunkTree!
        detail: f32,
        old_lod0_focus: PointN<Nf>,
        new_lod0_focus: PointN<Nf>,
        old_lod0_clip_extent: &ExtentN<Ni>,
        new_lod0_clip_extent: &ExtentN<Ni>,
        predicate: &mut impl FnMut(ChunkKey<Ni>) -> bool,
        event_rx: &mut impl FnMut(ClipEvent<Ni>),
    ) {
        if !predicate(node_key) {
            return;
        }

        let node_lod0_extent = self.indexer.chunk_extent_at_lower_lod(node_key, 0);
        let old_node_overlap = node_lod0_extent.intersection(old_lod0_clip_extent);
        let new_node_overlap = node_lod0_extent.intersection(new_lod0_clip_extent);
        let chunk_in_old_extent = !old_node_overlap.is_empty();
        let chunk_in_new_extent = !new_node_overlap.is_empty();

        if !chunk_in_old_extent && !chunk_in_new_extent {
            // There are no events for chunks that have not recently touched the clip extent.
            return;
        }

        if node_key.lod == 0 {
            // This node is active, and it can't be split or merged. But it might have just entered or exited.
            let entered = !chunk_in_old_extent && chunk_in_new_extent;
            if entered {
                event_rx(ClipEvent::Enter(node_key));
                return;
            }
            let exited = chunk_in_old_extent && !chunk_in_new_extent;
            if exited {
                event_rx(ClipEvent::Exit(node_key));
            }
            return;
        }

        if old_node_overlap == node_lod0_extent && new_node_overlap == node_lod0_extent {
            // This node was and is still a subset of the clip extent. In this case, enter and exit events are not possible.
            // We can now restrict ourselves to looking at occupied nodes.
            if let Some(child_mask) = self.get_child_mask(node_key) {
                self.occupied_clipmap_events_recursive(
                    node_key,
                    child_mask,
                    detail,
                    old_lod0_focus,
                    new_lod0_focus,
                    predicate,
                    event_rx,
                );
            }
            return;
        }

        // At this point we know the chunk only partially intersects the clip extent. All events are still possible.

        let old_normalized_dist = self.normalized_dist(old_lod0_focus, node_key);
        let new_normalized_dist = self.normalized_dist(new_lod0_focus, node_key);
        let was_active = old_normalized_dist > PointN::fill(detail);
        let is_active = new_normalized_dist > PointN::fill(detail);

        match (was_active, is_active) {
            // Old and new agree this chunk is not active.
            (false, false) => {
                // Just continue checking for events. We need to traverse all child orthants in case any descendants need to be
                // generated.
                for child_i in 0..PointN::NUM_CORNERS {
                    self.clipmap_events_recursive(
                        self.indexer.child_chunk_key(node_key, child_i),
                        detail,
                        old_lod0_focus,
                        new_lod0_focus,
                        old_lod0_clip_extent,
                        new_lod0_clip_extent,
                        predicate,
                        event_rx,
                    );
                }
                return;
            }
            (false, true) => {
                // This node just became active. Merge the children into this node.
                if let Some(child_mask) = self.get_child_mask(node_key) {
                    let old_chunks = self.collect_children(node_key, child_mask);
                    event_rx(ClipEvent::Merge(MergeChunks {
                        old_chunks,
                        new_chunk: node_key,
                    }));

                    // Don't generate more than one event per node.
                    return;
                }
            }
            (true, false) => {
                // This node just became inactive. Split this node into the children.
                if let Some(child_mask) = self.get_child_mask(node_key) {
                    let new_chunks = self.collect_children(node_key, child_mask);
                    event_rx(ClipEvent::Split(SplitChunk {
                        old_chunk: node_key,
                        new_chunks,
                    }));
                }
                return;
            }
            // Old and new agree this node is active. No need to merge or split.
            (true, true) => (),
        }

        // This node is active (based on early returns above). Check for enters and exits.
        let entered = !chunk_in_old_extent && chunk_in_new_extent;
        if entered {
            event_rx(ClipEvent::Enter(node_key));
            return;
        }
        let exited = chunk_in_old_extent && !chunk_in_new_extent;
        if exited {
            event_rx(ClipEvent::Exit(node_key));
        }
    }

    fn occupied_clipmap_events_recursive(
        &self,
        node_key: ChunkKey<Ni>, // Precondition: not LOD0.
        node_child_mask: u8,
        detail: f32,
        old_lod0_focus: PointN<Nf>,
        new_lod0_focus: PointN<Nf>,
        predicate: &mut impl FnMut(ChunkKey<Ni>) -> bool,
        event_rx: &mut impl FnMut(ClipEvent<Ni>),
    ) {
        let old_normalized_dist = self.normalized_dist(old_lod0_focus, node_key);
        let new_normalized_dist = self.normalized_dist(new_lod0_focus, node_key);
        let was_active = old_normalized_dist > PointN::fill(detail);
        let is_active = new_normalized_dist > PointN::fill(detail);

        match (was_active, is_active) {
            // Old and new agree this chunk is not active.
            (false, false) => {
                // Just continue checking for events.
                for child_i in 0..PointN::NUM_CORNERS {
                    if child_mask_has_child(node_child_mask, child_i) {
                        let child_key = self.indexer.child_chunk_key(node_key, child_i);
                        if !predicate(child_key) || child_key.lod == 0 {
                            continue;
                        }
                        let child_node_child_mask = self.get_child_mask(child_key).unwrap();
                        self.occupied_clipmap_events_recursive(
                            child_key,
                            child_node_child_mask,
                            detail,
                            old_lod0_focus,
                            new_lod0_focus,
                            predicate,
                            event_rx,
                        );
                    }
                }
            }
            (false, true) => {
                // This node just became active. Merge the children into this node.
                let old_chunks = self.collect_children(node_key, node_child_mask);
                event_rx(ClipEvent::Merge(MergeChunks {
                    old_chunks,
                    new_chunk: node_key,
                }));
            }
            (true, false) => {
                // This node just became inactive. Split this node into the children.
                let new_chunks = self.collect_children(node_key, node_child_mask);
                event_rx(ClipEvent::Split(SplitChunk {
                    old_chunk: node_key,
                    new_chunks,
                }));
            }
            // Old and new agree this node is active. No need to merge or split.
            (true, true) => (),
        }
    }

    fn collect_children(&self, parent_key: ChunkKey<Ni>, child_mask: u8) -> Vec<ChunkKey<Ni>> {
        (0..PointN::NUM_CORNERS)
            .filter_map(|child_i| {
                if child_mask_has_child(child_mask, child_i) {
                    Some(self.indexer.child_chunk_key(parent_key, child_i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculates the non-negative `D / S` metric used to determine our node-splitting condition.
    fn normalized_dist(&self, lod0_focus: PointN<Nf>, node_key: ChunkKey<Ni>) -> PointN<Nf> {
        let scale = (1 << node_key.lod) as f32;
        let chunk_shape = self.indexer.chunk_shape();
        let lod0_chunk_center = PointN::<Nf>::from(node_key.minimum + (chunk_shape >> 1)) * scale;
        let lod0_chunk_shape = PointN::<Nf>::from(chunk_shape) * scale;
        let dist_from_center = lod0_focus.l2_distance_squared(lod0_chunk_center).sqrt();
        // TODO: check for divide by zero
        PointN::fill(dist_from_center) / lod0_chunk_shape
    }
}

/// An event triggered by movement of the clipmap focus.
///
/// Note that while the `Split` and `Merge` events only occur for *occupied* chunk slots (those with voxel data), `Enter` and
/// `Exit` events can occur for occupied *or* vacant chunk slots. This is at least necessary for detecting slots that need to
/// have data loaded or generated on an `Enter` event. It is less necessary for evicting data on `Exit` events, but it's easy
/// enough to ignore `Exit` events for vacant chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClipEvent<N> {
    /// A new chunk slot entered the clip extent this frame, and it should be rendered at the given LOD.
    Enter(ChunkKey<N>),
    /// A chunk slot exited the clip extent this frame. It was previously rendered at the given LOD.
    Exit(ChunkKey<N>),
    /// The desired sample rate for this chunk increased this frame.
    Split(SplitChunk<N>),
    /// The desired sample rate for this chunk decreased this frame.
    Merge(MergeChunks<N>),
}

/// A 2-dimensional `ClipEvent`.
pub type ClipEvent2 = ClipEvent<[i32; 2]>;
/// A 3-dimensional `ClipEvent`.
pub type ClipEvent3 = ClipEvent<[i32; 3]>;

/// Split `old_chunk` into many `new_chunks`. The number of new chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitChunk<N> {
    pub old_chunk: ChunkKey<N>,
    pub new_chunks: Vec<ChunkKey<N>>,
}

/// Merge many `old_chunks` into `new_chunk`. The number of old chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeChunks<N> {
    pub old_chunks: Vec<ChunkKey<N>>,
    pub new_chunk: ChunkKey<N>,
}
