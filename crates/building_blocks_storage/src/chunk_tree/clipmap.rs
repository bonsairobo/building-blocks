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

        if node_key.lod == 0 {
            if node_bounded_by_clip_sphere {
                // This node is active!
                active_rx(node_key);
            }
            return;
        }

        // Normalize distance by chunk shape.
        let norm_dist = dist / node_lod0_radius;
        let is_active = norm_dist > detail;

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

    /// Like `active_clipmap_chunks`, but it detects [`ClipEvent`]s triggered by movement of the focal point from
    /// `old_clip_sphere` to `new_clip_sphere`.
    ///
    /// When detecting enter/exit events, `enter_exit_min_lod` will be used to stop the search earlier than LOD0. This
    /// significantly improves performance for the branching and bounding algorithm. When configured, the user must assume that
    /// if an enter/exit event is received, it also applies to all descendants of the given chunk key.
    pub fn clipmap_events(
        &self,
        detail: f32,
        enter_exit_min_lod: u8,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        mut event_rx: impl FnMut(ClipEvent<Ni>),
    ) {
        assert!(old_clip_sphere.radius > 0.0);
        assert!(new_clip_sphere.radius > 0.0);

        let root_lod = self.root_lod();

        let old_lod0_clip_extent = extent_containing_sphere(old_clip_sphere);
        let new_lod0_clip_extent = extent_containing_sphere(new_clip_sphere);
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

        // Optimization: only calculate bounding sphere radii once. Use squared radius to avoid calling sqrt later.
        let lod0_chunk_shape_max_comp = self.chunk_shape().max_component();
        let num_lods = root_lod as usize + 1;
        let chunk_bounding_radii: Vec<_> = (0..num_lods)
            .map(|lod| ((lod0_chunk_shape_max_comp >> 1) << lod) as f32 * 3f32.sqrt())
            .collect();

        for chunk_min in root_nodes.into_iter() {
            self.clipmap_events_recursive(
                ChunkKey::new(root_lod, chunk_min),
                detail,
                enter_exit_min_lod,
                old_clip_sphere,
                new_clip_sphere,
                &chunk_bounding_radii,
                false,
                false,
                &mut event_rx,
            );
        }
    }

    fn clipmap_events_recursive(
        &self,
        node_key: ChunkKey<Ni>, // May not exist in the ChunkTree!
        detail: f32,
        enter_exit_min_lod: u8,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        chunk_bounding_radii: &[f32],
        ancestor_was_active: bool,
        ancestor_is_active: bool,
        event_rx: &mut impl FnMut(ClipEvent<Ni>),
    ) {
        let node_lod0_extent = self.indexer.chunk_extent_at_lower_lod(node_key, 0);
        let node_lod0_center =
            PointN::<Nf>::from(node_lod0_extent.minimum + (node_lod0_extent.shape >> 1));
        let node_lod0_radius = chunk_bounding_radii[node_key.lod as usize];

        // Calculate the Euclidean distance from each focus the center of the chunk.
        let old_dist = old_clip_sphere
            .center
            .l2_distance_squared(node_lod0_center)
            .sqrt();
        let new_dist = new_clip_sphere
            .center
            .l2_distance_squared(node_lod0_center)
            .sqrt();

        let node_intersects_old_clip_sphere = old_dist - node_lod0_radius < old_clip_sphere.radius;
        let node_intersects_new_clip_sphere = new_dist - node_lod0_radius < new_clip_sphere.radius;

        if !node_intersects_old_clip_sphere && !node_intersects_new_clip_sphere {
            // There are no events for this node or any of its descendants.
            return;
        }

        let node_bounded_by_old_clip_sphere = old_dist + node_lod0_radius < old_clip_sphere.radius;
        let node_bounded_by_new_clip_sphere = new_dist + node_lod0_radius < new_clip_sphere.radius;
        let stable_bounded = node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere;

        // We can't have more active enter/exits in this tree, so it might be OK to skip the inactive ones.
        let no_active_enter_exits = ancestor_was_active && ancestor_is_active;

        if (no_active_enter_exits && node_key.lod < enter_exit_min_lod) || stable_bounded {
            // This node was and is still bounded by the clip sphere. In this case, enter and exit events are not possible
            // in this tree. We can now restrict ourselves to looking for split/merge events on occupied nodes.
            if node_key.lod == 0 || ancestor_was_active || ancestor_is_active {
                // Merge/split can't happen here if an ancestor is active.
                return;
            }
            if stable_bounded {
                if let Some(child_mask) = self.get_child_mask(node_key) {
                    self.only_detect_split_or_merge_recursive(
                        node_key,
                        child_mask,
                        detail,
                        old_clip_sphere,
                        new_clip_sphere,
                        chunk_bounding_radii,
                        event_rx,
                    );
                }
            }
            return;
        }

        let old_norm_dist = old_dist / node_lod0_radius;
        let new_norm_dist = new_dist / node_lod0_radius;
        let was_active = !ancestor_was_active && (node_key.lod == 0 || old_norm_dist > detail);
        let is_active = !ancestor_is_active && (node_key.lod == 0 || new_norm_dist > detail);

        // Recurse.
        if node_key.lod > 0 {
            // Just continue checking for all events. All descendants, occupied or vacant, are contenders.
            for child_i in 0..PointN::NUM_CORNERS {
                self.clipmap_events_recursive(
                    self.indexer.child_chunk_key(node_key, child_i),
                    detail,
                    enter_exit_min_lod,
                    old_clip_sphere,
                    new_clip_sphere,
                    chunk_bounding_radii,
                    ancestor_was_active || was_active,
                    ancestor_is_active || is_active,
                    event_rx,
                );
            }
        }

        // We check for enter/exit events after recursing because we need to guarantee all descendant enter/exit events come
        // first.
        if !node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere {
            event_rx(ClipEvent::Enter(node_key, is_active));
        } else if node_bounded_by_old_clip_sphere && !node_bounded_by_new_clip_sphere {
            event_rx(ClipEvent::Exit(node_key, was_active));
        }
    }

    fn only_detect_split_or_merge_recursive(
        &self,
        node_key: ChunkKey<Ni>, // Precondition: not LOD0.
        node_child_mask: u8,
        detail: f32,
        old_clip_sphere: Sphere<Nf>,
        new_clip_sphere: Sphere<Nf>,
        chunk_bounding_radii: &[f32],
        event_rx: &mut impl FnMut(ClipEvent<Ni>),
    ) {
        let node_lod0_extent = self.indexer.chunk_extent_at_lower_lod(node_key, 0);
        let lod0_chunk_center =
            PointN::<Nf>::from(node_lod0_extent.minimum + (node_lod0_extent.shape >> 1));
        let node_lod0_radius = chunk_bounding_radii[node_key.lod as usize];

        // Calculate the Euclidean distance from each focus the center of the chunk.
        let old_dist = old_clip_sphere
            .center
            .l2_distance_squared(lod0_chunk_center)
            .sqrt();
        let new_dist = new_clip_sphere
            .center
            .l2_distance_squared(lod0_chunk_center)
            .sqrt();

        let old_norm_dist = old_dist / node_lod0_radius;
        let new_norm_dist = new_dist / node_lod0_radius;

        let was_active = old_norm_dist > detail;
        let is_active = new_norm_dist > detail;

        match (was_active, is_active) {
            // Old and new frames agree this chunk is not active.
            (false, false) => {
                // Keep looking for any active descendants.
                if node_key.lod > 1 {
                    for child_i in 0..PointN::NUM_CORNERS {
                        if child_mask_has_child(node_child_mask, child_i) {
                            let child_key = self.indexer.child_chunk_key(node_key, child_i);
                            let child_node_child_mask = self.get_child_mask(child_key).unwrap();
                            self.only_detect_split_or_merge_recursive(
                                child_key,
                                child_node_child_mask,
                                detail,
                                old_clip_sphere,
                                new_clip_sphere,
                                chunk_bounding_radii,
                                event_rx,
                            );
                        }
                    }
                }
            }
            (false, true) => {
                // This node just became active, and none if its ancestors were active, so it must have active descendants.
                // Merge those active descendants into this node.
                let mut old_chunks = Vec::with_capacity(8);
                for child_i in 0..PointN::NUM_CORNERS {
                    if child_mask_has_child(node_child_mask, child_i) {
                        let child_key = self.indexer.child_chunk_key(node_key, child_i);
                        self.clipmap_active_chunks_recursive(
                            child_key,
                            detail,
                            old_clip_sphere,
                            &mut |active_chunk| old_chunks.push(active_chunk),
                        );
                    }
                }
                event_rx(ClipEvent::Merge(MergeChunks {
                    old_chunks,
                    new_chunk: node_key,
                }));
            }
            (true, false) => {
                // This node just became inactive, and none of its ancestors were active, so it must have active descendants.
                // Split this node into the children.
                let mut new_chunks = Vec::with_capacity(8);
                for child_i in 0..PointN::NUM_CORNERS {
                    if child_mask_has_child(node_child_mask, child_i) {
                        let child_key = self.indexer.child_chunk_key(node_key, child_i);
                        self.clipmap_active_chunks_recursive(
                            child_key,
                            detail,
                            new_clip_sphere,
                            &mut |active_chunk| new_chunks.push(active_chunk),
                        );
                    }
                }
                event_rx(ClipEvent::Split(SplitChunk {
                    old_chunk: node_key,
                    new_chunks,
                }));
            }
            // Old and new agree this node is active. No need to merge or split. None of the descendants can merge or split
            // either.
            (true, true) => (),
        }
    }
}

fn extent_containing_sphere<Nf, Ni>(sphere: Sphere<Nf>) -> ExtentN<Ni>
where
    PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
{
    let aabb = ExtentN::from_min_and_shape(
        PointN::fill(-sphere.radius),
        PointN::fill(2.0 * sphere.radius),
    ) + sphere.center;
    aabb.containing_integer_extent()
}

/// An event triggered by movement of the clipmap focus. A given chunk slot is guaranteed to receive at most one event per
/// frame.
///
/// Note that while the `Split` and `Merge` events only occur for *occupied* chunk slots (those with voxel data), `Enter` and
/// `Exit` events can occur for occupied *or* vacant chunk slots. This is at least necessary for detecting slots that need to
/// have data loaded or generated on an `Enter` event. It is less necessary for evicting data on `Exit` events, but it's easy
/// enough to ignore `Exit` events for vacant chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClipEvent<N> {
    /// A new chunk slot entered the clip extent this frame, meaning it is now bounded by the clip sphere. The `bool` is `true`
    /// iff the chunk slot is now active.
    ///
    /// This event upholds an essential ordering guarantee; if such an event is received for some chunk, then all descendant
    /// chunks must already be bounded by the clip sphere, i.e. they've already entered and have not since exited. This means if
    /// you always generate chunks when they enter the clip sphere (in event order), then it is safe to assume all descendants
    /// are ready to be used, e.g. for downsampling.
    Enter(ChunkKey<N>, bool),
    /// A chunk slot that was bounded by the clip sphere in the previous frame is no longer bounded in this frame. The `bool` is
    /// `true` iff the chunk slot was active before exiting.
    ///
    /// This is the reverse condition of the `Enter` event, meaning that chunks should enter and exit the clip sphere at the
    /// same distance from the focus.
    ///
    /// For a given chunk, its sequence of `Enter` and `Exit` events will always look like
    /// ```text
    ///  ..., Enter, Exit, Enter, Exit, ...
    /// ```
    /// so you can manage any resources tied to a chunk be creating them on `Enter` and freeing them on `Exit`.
    Exit(ChunkKey<N>, bool),
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

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn old_and_new_clip_sphere_no_overlap_without_data() {
        let chunks = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
            chunk_shape: Point3i::fill(16),
            ambient_value: 0,
            root_lod: 5,
        })
        .build_with_hash_map_storage();

        let old_clip_sphere = Sphere3 {
            center: Point3::ZERO,
            radius: 100.0,
        };
        let new_clip_sphere = Sphere3 {
            center: Point3f::fill(300.0),
            radius: 100.0,
        };

        let mut lod0_entrances = HashSet::new();
        let mut lod0_exits = HashSet::new();
        chunks.clipmap_events(
            6.0,
            0,
            old_clip_sphere,
            new_clip_sphere,
            |event| match event {
                ClipEvent::Enter(key, _) => {
                    if key.lod == 0 {
                        lod0_entrances.insert(key);
                    }
                }
                ClipEvent::Exit(key, _) => {
                    if key.lod == 0 {
                        lod0_exits.insert(key);
                    }
                }
                _ => panic!("unexpected event"),
            },
        );

        assert!(!lod0_entrances.is_empty());
        assert!(!lod0_exits.is_empty());

        // All of the chunk in the old sphere should have exited.
        let expected_lod0_exits: HashSet<_> = chunks
            .indexer
            .chunk_mins_for_extent(&extent_containing_sphere(old_clip_sphere))
            .map(|chunk_min| ChunkKey3::new(0, chunk_min))
            .filter(|key| {
                sphere_contains_extent(
                    old_clip_sphere,
                    chunks.indexer.extent_for_chunk_with_min(key.minimum),
                )
            })
            .collect();
        // All of the chunk in the new sphere should have entered.
        let expected_lod0_entrances: HashSet<_> = chunks
            .indexer
            .chunk_mins_for_extent(&extent_containing_sphere(new_clip_sphere))
            .map(|chunk_min| ChunkKey3::new(0, chunk_min))
            .filter(|key| {
                sphere_contains_extent(
                    new_clip_sphere,
                    chunks.indexer.extent_for_chunk_with_min(key.minimum),
                )
            })
            .collect();

        assert!(!expected_lod0_exits.is_empty());
        assert!(!expected_lod0_entrances.is_empty());

        assert!(
            expected_lod0_entrances.eq(&lod0_entrances),
            "# expected = {:?} # missing = {:?}",
            expected_lod0_entrances.len(),
            expected_lod0_entrances.difference(&lod0_entrances).count()
        );
        assert!(
            expected_lod0_exits.eq(&lod0_exits),
            "# expected = {:?} # missing = {:?}",
            expected_lod0_exits.len(),
            expected_lod0_exits.difference(&lod0_exits).count()
        );
    }

    fn sphere_contains_extent(sphere: Sphere3, extent: Extent3i) -> bool {
        let max_comp = extent.shape.max_component();
        let chunk_bounding_radius = (max_comp >> 1) as f32 * 3f32.sqrt();
        let extent_center = Point3f::from(extent.minimum + (extent.shape >> 1));
        let dist = sphere.center.l2_distance_squared(extent_center).sqrt();
        dist + chunk_bounding_radius < sphere.radius
    }
}
