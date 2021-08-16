use crate::{chunk::ChunkNode, dev_prelude::*};

use building_blocks_core::prelude::*;

impl<Ni, Nf, T, Usr, Bldr, Store> ChunkTree<Ni, T, Bldr, Store>
where
    PointN<Ni>: IntegerPoint<Ni, FloatPoint = PointN<Nf>>,
    PointN<Nf>: Distance + From<PointN<Ni>> + Point<Scalar = f32>,
    Bldr: ChunkTreeBuilder<Ni, T, Chunk = Usr>,
    Store: ChunkStorage<Ni, Chunk = ChunkNode<Usr>> + for<'r> IterChunkKeys<'r, Ni>,
{
    /// Traverses from all roots to find the currently "active" chunks. Active chunks are passed to the `active_rx` callback.
    ///
    /// By "active," we mean that, given the current location of `lod0_center` (e.g. a camera), the chunk has the desired LOD
    /// for rendering. More specifically, let `D` be the Euclidean distance from `lod0_center` to the center of the chunk (in
    /// LOD0 space), and let `S` be the shape of the chunk (in LOD0 space). The chunk cannot be active (must be "split") if `(S
    /// / D) > clip_radius`. Along a given path from a root chunk to a leaf, the least detailed chunk that *can* be active is
    /// used. `active_rx` will also receive the value `S / D` as it may be useful for continuous LOD blending.
    ///
    /// `predicate` is used to terminate traversal of a subtree early so that the rejected chunk and any of its descendants
    /// cannot be considered active. This could be useful for e.g. culling chunks outside of a view frustum.
    pub fn clipmap_active_chunks(
        &self,
        clip_radius: f32,
        lod0_center: PointN<Nf>,
        mut predicate: impl FnMut(ChunkKey<Ni>) -> bool,
        mut active_rx: impl FnMut(ChunkKey<Ni>, PointN<Nf>),
    ) {
        let root_lod = self.root_lod();
        let root_storage = self.lod_storage(root_lod);
        for &chunk_min in root_storage.chunk_keys() {
            self.clipmap_active_chunks_recursive(
                ChunkKey::new(root_lod, chunk_min),
                clip_radius,
                lod0_center,
                &mut predicate,
                &mut active_rx,
            );
        }
    }

    fn clipmap_active_chunks_recursive(
        &self,
        node_key: ChunkKey<Ni>,
        clip_radius: f32,
        lod0_center: PointN<Nf>,
        predicate: &mut impl FnMut(ChunkKey<Ni>) -> bool,
        active_rx: &mut impl FnMut(ChunkKey<Ni>, PointN<Nf>),
    ) {
        if !predicate(node_key) {
            return;
        }

        let node = self.get_chunk_node(node_key).unwrap();
        let apparent_shape = self.apparent_shape(lod0_center, node_key);
        if node_key.lod > 0 && apparent_shape >= PointN::fill(clip_radius) {
            // Need to split, continue by recursing on the children.
            for child_i in 0..PointN::NUM_CORNERS {
                if node.has_child(child_i) {
                    self.clipmap_active_chunks_recursive(
                        self.indexer.child_chunk_key(node_key, child_i),
                        clip_radius,
                        lod0_center,
                        predicate,
                        active_rx,
                    );
                }
            }
        } else {
            // This node is active!
            active_rx(node_key, apparent_shape);
        }
    }

    /// Like `active_clipmap_chunks`, but it only detects changes in the set of active chunks after the focal point moves from
    /// `old_lod0_center` to `new_lod0_center`.
    ///
    /// `update_rx` will receive both the split/merge update as well as the `S / D` value of the newly active chunk (children of
    /// the split or parent of the merge).
    pub fn clipmap_updates(
        &self,
        clip_radius: f32,
        old_lod0_center: PointN<Nf>,
        new_lod0_center: PointN<Nf>,
        mut predicate: impl FnMut(ChunkKey<Ni>) -> bool,
        mut update_rx: impl FnMut(LodChunkUpdate<Ni>, PointN<Nf>),
    ) {
        let root_lod = self.root_lod();
        let root_storage = self.lod_storage(root_lod);
        for &chunk_min in root_storage.chunk_keys() {
            self.clipmap_updates_recursive(
                ChunkKey::new(root_lod, chunk_min),
                clip_radius,
                old_lod0_center,
                new_lod0_center,
                &mut predicate,
                &mut update_rx,
            );
        }
    }

    fn clipmap_updates_recursive(
        &self,
        node_key: ChunkKey<Ni>,
        clip_radius: f32,
        old_lod0_center: PointN<Nf>,
        new_lod0_center: PointN<Nf>,
        predicate: &mut impl FnMut(ChunkKey<Ni>) -> bool,
        update_rx: &mut impl FnMut(LodChunkUpdate<Ni>, PointN<Nf>),
    ) {
        if !predicate(node_key) || node_key.lod == 0 {
            return;
        }

        let node = self.get_chunk_node(node_key).unwrap();

        let old_apparent_shape = self.apparent_shape(old_lod0_center, node_key);
        let new_apparent_shape = self.apparent_shape(new_lod0_center, node_key);

        let old_was_split = old_apparent_shape >= PointN::fill(clip_radius);
        let new_is_split = new_apparent_shape >= PointN::fill(clip_radius);

        match (old_was_split, new_is_split) {
            (true, true) => {
                // Old and new agree this node is not active. Just continue checking for updates.
                for child_i in 0..PointN::NUM_CORNERS {
                    if node.has_child(child_i) {
                        self.clipmap_updates_recursive(
                            self.indexer.child_chunk_key(node_key, child_i),
                            clip_radius,
                            old_lod0_center,
                            new_lod0_center,
                            predicate,
                            update_rx,
                        );
                    }
                }
            }
            (true, false) => {
                // Merge the children into this node.
                let old_chunks = self.collect_children(node_key, node);
                update_rx(
                    LodChunkUpdate::Merge(MergeChunks {
                        old_chunks,
                        new_chunk: node_key,
                    }),
                    new_apparent_shape,
                );
            }
            (false, true) => {
                // Split this node into the children.
                let new_chunks = self.collect_children(node_key, node);
                update_rx(
                    LodChunkUpdate::Split(SplitChunk {
                        old_chunk: node_key,
                        new_chunks,
                    }),
                    new_apparent_shape,
                );
            }
            (false, false) => {
                // Old and new agree this node is active. No update.
                return;
            }
        }
    }

    fn collect_children(
        &self,
        parent_key: ChunkKey<Ni>,
        node: &ChunkNode<Usr>,
    ) -> Vec<ChunkKey<Ni>> {
        (0..PointN::NUM_CORNERS)
            .filter_map(|child_i| {
                if node.has_child(child_i) {
                    Some(self.indexer.child_chunk_key(parent_key, child_i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculates the metric used to determine our node-splitting condition.
    fn apparent_shape(&self, lod0_center: PointN<Nf>, node_key: ChunkKey<Ni>) -> PointN<Nf> {
        let scale = (1 << node_key.lod) as f32;
        let chunk_shape = self.indexer.chunk_shape();
        let lod0_chunk_center = PointN::<Nf>::from(node_key.minimum + (chunk_shape >> 1)) * scale;
        let lod0_chunk_shape = PointN::<Nf>::from(chunk_shape) * scale;
        let dist_from_center = lod0_center.l2_distance_squared(lod0_chunk_center).sqrt();
        // TODO: check for divide by zero
        lod0_chunk_shape / dist_from_center
    }
}

/// A notification that a chunk (at a particular level of detail) must be split or merged. This is usually the result of a
/// camera movement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodChunkUpdate<N> {
    Split(SplitChunk<N>),
    Merge(MergeChunks<N>),
}

/// A 2-dimensional `LodChunkUpdate`.
pub type LodChunkUpdate2 = LodChunkUpdate<[i32; 2]>;
/// A 3-dimensional `LodChunkUpdate`.
pub type LodChunkUpdate3 = LodChunkUpdate<[i32; 3]>;

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
