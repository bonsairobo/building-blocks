use super::StateBit;

use crate::dev_prelude::*;

use building_blocks_core::{prelude::*, Sphere};

use float_ord::FloatOrd;
use std::collections::BinaryHeap;

impl<Ni, Nf, T, Usr, Bldr, Store> ChunkTree<Ni, T, Bldr, Store>
where
    PointN<Ni>: std::hash::Hash + IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
    Bldr: ChunkTreeBuilder<Ni, T, Chunk = Usr>,
    Store: ChunkStorage<Ni, Chunk = Usr> + for<'r> IterChunkKeys<'r, Ni>,
{
    /// Sets all of the `child_needs_loading` bits on the node at `key`, creating a new node if one does not exist.
    ///
    /// Ancestor nodes will not be touched, so it is advisable to start searching at `key` when finding chunks to load.
    pub fn clipmap_mark_node_for_loading_if_vacant(&mut self, key: ChunkKey<Ni>) {
        let mut already_exists = true;
        let state =
            self.storages[key.lod as usize].get_mut_node_state_or_insert_with(key.minimum, || {
                already_exists = false;
                ChunkNode::new_empty()
            });

        if already_exists {
            return;
        }

        state.child_needs_loading_bits.set_all();

        self.link_node(key);
    }

    /// Inserts `chunk` at `key` and unsets the `child_needs_loading` bit on the parent node. Ancestor nodes also have their
    /// bits unset if applicable.
    pub fn clipmap_write_loaded_chunk(&mut self, mut key: ChunkKey<Ni>, chunk: Option<Usr>) {
        if let Some(chunk) = chunk {
            self.write_chunk(key, chunk);
        }

        while key.lod < self.root_lod() {
            let parent = self.indexer.parent_chunk_key(key);
            let corner_index = self.indexer.corner_index(key.minimum);
            let (parent_state, _) = self.get_mut_node_state(parent).unwrap();
            parent_state
                .child_needs_loading_bits
                .unset_bit(corner_index);
            if !parent_state.child_needs_loading_bits.all() {
                break;
            }
            key = parent;
        }
    }

    /// Detects changes in desired sample rate for occupied chunks based on distance from an observer at the center of
    /// `clip_sphere`.
    ///
    /// Along any path from a leaf chunk to the root, there must be exactly one *active* chunk. By "active," we mean that, given
    /// the current location of `clip_sphere`:
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
    /// The chunk is a *render candidate* if
    ///
    /// ```text
    ///     D + B < clip_sphere.radius && (D / S) > detail
    /// ```
    ///
    /// where `detail` is a nonnegative constant parameter supplied by you. Along a given path from a leaf to the root, only the
    /// least detailed render candidate chunk is active.
    ///
    /// In order to prioritize rendering chunks closer to the observer, this method will traverse the tree, starting close to
    /// the center of `clip_sphere` and working outwards. The `ChunkTree` will remember which chunks are active, and if we
    /// detect a chunk that should either increase or decrease sample rate, then it should be either `Split` or `Merge`d
    /// respectively.
    ///
    /// A `Merge` only activates a single new chunk, so it counts for a single unit from the `render_chunk_budget`. A `Split`
    /// may activate arbitrarily many new chunks depending on how much the detail needs to change, and it will add them up while
    /// not exceeding a total of `render_chunk_budget` for the entire function call.
    pub fn clipmap_render_updates(
        &self,
        detail: f32,
        clip_sphere: Sphere<Nf>,
        render_chunk_budget: usize,
        mut rx: impl FnMut(LodChange<Ni>),
    ) {
        assert!(clip_sphere.radius > 0.0);

        // A map from descendant key to ancestor key. These are descendants of a chunk that we've committed to splitting. They
        // might also be in the candidate heap awaiting another potential split. If one gets split again, it'll need to be
        // replaced in the map with its children. When traversal finishes, these will be committed as-is along with their split
        // ancestor.
        let mut committed_split_descendants = SmallKeyHashMap::new();

        let mut candidate_heap = BinaryHeap::new();
        let mut num_render_chunks = 0;

        self.visit_root_keys(|root| {
            candidate_heap.push(ChunkSphere::new(clip_sphere, &self.indexer, root));
        });

        while let Some(ChunkSphere {
            key: node_key,
            bounding_sphere: node_sphere,
            center_dist_to_observer,
            closest_dist_to_observer,
        }) = candidate_heap.pop()
        {
            let node_intersects_clip_sphere = closest_dist_to_observer < clip_sphere.radius;
            if !node_intersects_clip_sphere {
                continue;
            }

            let node_state = self.get_node_state(node_key).unwrap();

            let was_active = node_state.state_bits.bit_is_set(StateBit::Render as u8);
            let is_active =
                node_key.lod == 0 || center_dist_to_observer / node_sphere.radius > detail;

            if is_active {
                node_state.state_bits.set_bit(StateBit::Render as u8);
            } else {
                node_state.state_bits.unset_bit(StateBit::Render as u8);
            }

            match (was_active, is_active) {
                // Old and new frames agree this chunk is not active.
                (false, false) => {
                    // Keep looking for any active descendants.
                    self.visit_child_keys_of_node(node_key, &node_state, |child_key| {
                        candidate_heap.push(ChunkSphere::new(
                            clip_sphere,
                            &self.indexer,
                            child_key,
                        ));
                    });
                }
                (false, true) => {
                    // This node just became active, and none if its ancestors were active, so it must have active descendants.
                    // Merge those active descendants into this node.
                    let mut old_chunks = Vec::with_capacity(8);
                    if node_key.lod > 0 {
                        self.visit_child_keys_of_node(node_key, &node_state, |child_key| {
                            self.visit_tree_keys(child_key, |descendant_key| {
                                let descendant_node = self.get_node(descendant_key).unwrap();
                                let descendant_was_active = descendant_node
                                    .state
                                    .state_bits
                                    .fetch_and_unset_bit(StateBit::Render as u8);
                                if descendant_was_active {
                                    old_chunks.push(descendant_key);
                                }
                                !descendant_was_active
                            });
                        });
                    }

                    rx(LodChange::Merge(MergeChunks {
                        old_chunks,
                        new_chunk: node_key,
                    }));
                    num_render_chunks += 1;
                }
                (true, false) => {
                    // This node just became inactive, and none of its ancestors were active, so it must have active
                    // descendants. Split this node into active descendants.

                    // This node might have already split off from some ancestor.
                    let split_ancestor =
                        if let Some(a) = committed_split_descendants.remove(&node_key) {
                            num_render_chunks -= 1;
                            a
                        } else {
                            node_key
                        };

                    // This chunk could potentially need to be split by multiple levels, but we need to be careful not to run
                    // out of render chunk budget. To be fair to other chunks in the queue that need to be split, we will only
                    // split by one layer for now, but we'll re-insert the children back into the heap so they can be split
                    // again if we still have budget.
                    self.visit_child_keys_of_node(node_key, &node_state, |child_key| {
                        let child_node = self.get_node(child_key).unwrap();
                        child_node.state.state_bits.set_bit(StateBit::Render as u8);

                        num_render_chunks += 1;
                        committed_split_descendants.insert(child_key, split_ancestor);
                        if child_key.lod > 0 {
                            candidate_heap.push(ChunkSphere::new(
                                clip_sphere,
                                &self.indexer,
                                child_key,
                            ));
                        }
                    });
                }
                // Old and new agree this node is active. No need to merge or split. None of the descendants can merge or split
                // either.
                (true, true) => (),
            }

            if num_render_chunks >= render_chunk_budget {
                break;
            }
        }

        // Reconstruct the splits and send to the receiver.
        let mut splits = SmallKeyHashMap::new();
        for (descendant, split_ancestor) in committed_split_descendants.into_iter() {
            let split = splits.entry(split_ancestor).or_insert_with(|| SplitChunk {
                old_chunk: split_ancestor,
                new_chunks: Vec::with_capacity(8),
            });
            split.new_chunks.push(descendant);
        }

        for (_, split) in splits.into_iter() {
            rx(LodChange::Split(split));
        }
    }
}

/// Detects new chunk slots at `detect_lod` that entered `new_clip_sphere` after it moved from `old_clip_sphere`.
pub fn clipmap_new_chunks<Ni, Nf>(
    indexer: &ChunkIndexer<Ni>,
    root_lod: u8,
    detect_lod: u8,
    detail: f32,
    old_clip_sphere: Sphere<Nf>,
    new_clip_sphere: Sphere<Nf>,
    mut rx: impl FnMut(ClipmapSlot<Ni>),
) where
    PointN<Ni>: std::hash::Hash + IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
{
    assert!(old_clip_sphere.radius > 0.0);
    assert!(new_clip_sphere.radius > 0.0);

    let new_lod0_clip_extent = new_clip_sphere.aabb().containing_integer_extent();
    let new_root_clip_extent =
        indexer.covering_ancestor_extent(new_lod0_clip_extent, root_lod as i32);

    for chunk_min in indexer.chunk_mins_for_extent(&new_root_clip_extent) {
        clipmap_new_chunks_recursive(
            indexer,
            ChunkKey::new(root_lod, chunk_min),
            detect_lod,
            detail,
            old_clip_sphere,
            new_clip_sphere,
            &mut rx,
        );
    }
}

fn clipmap_new_chunks_recursive<Ni, Nf>(
    indexer: &ChunkIndexer<Ni>,
    node_key: ChunkKey<Ni>, // May not exist in the ChunkTree!
    detect_lod: u8,
    detail: f32,
    old_clip_sphere: Sphere<Nf>,
    new_clip_sphere: Sphere<Nf>,
    rx: &mut impl FnMut(ClipmapSlot<Ni>),
) where
    PointN<Ni>: std::hash::Hash + IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
{
    let node_sphere = chunk_lod0_bounding_sphere(indexer, node_key);

    // Calculate the Euclidean distance from each focus the center of the chunk.
    let dist_to_old_clip_sphere = old_clip_sphere
        .center
        .l2_distance_squared(node_sphere.center)
        .sqrt();
    let dist_to_new_clip_sphere = new_clip_sphere
        .center
        .l2_distance_squared(node_sphere.center)
        .sqrt();

    let node_intersects_old_clip_sphere =
        dist_to_old_clip_sphere - node_sphere.radius < old_clip_sphere.radius;
    let node_intersects_new_clip_sphere =
        dist_to_new_clip_sphere - node_sphere.radius < new_clip_sphere.radius;

    if !node_intersects_old_clip_sphere && !node_intersects_new_clip_sphere {
        // There are no events for this node or any of its descendants.
        return;
    }

    let node_bounded_by_old_clip_sphere =
        dist_to_old_clip_sphere + node_sphere.radius < old_clip_sphere.radius;
    let node_bounded_by_new_clip_sphere =
        dist_to_new_clip_sphere + node_sphere.radius < new_clip_sphere.radius;

    if node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere {
        // This node is stably bounded, so enter events are not possible.
        return;
    }

    if node_key.lod > detect_lod {
        for child_i in 0..PointN::NUM_CORNERS {
            clipmap_new_chunks_recursive(
                indexer,
                indexer.child_chunk_key(node_key, child_i),
                detect_lod,
                detail,
                old_clip_sphere,
                new_clip_sphere,
                rx,
            );
        }
    } else {
        // This is the LOD where we want to detect entrances into the clip sphere.
        let is_render_candidate =
            node_key.lod == 0 || dist_to_new_clip_sphere / node_sphere.radius > detail;

        if !node_bounded_by_old_clip_sphere && node_bounded_by_new_clip_sphere {
            rx(ClipmapSlot {
                key: node_key,
                dist: dist_to_new_clip_sphere,
                is_render_candidate,
            });
        }
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

pub struct ClipmapSlot<N> {
    pub key: ChunkKey<N>,
    pub dist: f32,
    pub is_render_candidate: bool,
}

/// A 2-dimensional `ClipmapSlot`.
pub type ClipmapSlot2 = ClipmapSlot<[i32; 2]>;
/// A 3-dimensional `ClipmapSlot`.
pub type ClipmapSlot3 = ClipmapSlot<[i32; 3]>;

/// A chunk's desired sample rate has changed based on proximity to the center of the clip sphere.
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
pub struct SplitChunk<N> {
    pub old_chunk: ChunkKey<N>,
    pub new_chunks: Vec<ChunkKey<N>>,
}

/// Merge many `old_chunks` into `new_chunk`. The number of old chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeChunks<N> {
    pub old_chunks: Vec<ChunkKey<N>>,
    pub new_chunk: ChunkKey<N>,
}

#[derive(Clone)]
struct ChunkSphere<Ni, Nf> {
    bounding_sphere: Sphere<Nf>,
    key: ChunkKey<Ni>,
    center_dist_to_observer: f32,
    closest_dist_to_observer: f32,
}

impl<Ni, Nf> ChunkSphere<Ni, Nf>
where
    PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
{
    fn new(clip_sphere: Sphere<Nf>, indexer: &ChunkIndexer<Ni>, key: ChunkKey<Ni>) -> Self {
        let bounding_sphere = chunk_lod0_bounding_sphere(indexer, key);

        let center_dist_to_observer = clip_sphere
            .center
            .l2_distance_squared(bounding_sphere.center)
            .sqrt();
        // Subtract the bounding sphere's radius to estimate the distance from the observer to the *closest point* on the chunk.
        // This should make it more fair for higher LODs.
        let closest_dist_to_observer = center_dist_to_observer - bounding_sphere.radius;

        Self {
            bounding_sphere,
            key,
            center_dist_to_observer,
            closest_dist_to_observer,
        }
    }
}

impl<Ni, Nf> PartialEq for ChunkSphere<Ni, Nf>
where
    PointN<Ni>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<Ni, Nf> Eq for ChunkSphere<Ni, Nf> where PointN<Ni>: Eq {}

impl<Ni, Nf> PartialOrd for ChunkSphere<Ni, Nf>
where
    PointN<Ni>: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        FloatOrd(self.closest_dist_to_observer)
            .partial_cmp(&FloatOrd(other.closest_dist_to_observer))
            .map(|o| o.reverse())
    }
}

impl<Ni, Nf> Ord for ChunkSphere<Ni, Nf>
where
    PointN<Ni>: Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        FloatOrd(self.closest_dist_to_observer)
            .cmp(&FloatOrd(other.closest_dist_to_observer))
            .reverse()
    }
}
