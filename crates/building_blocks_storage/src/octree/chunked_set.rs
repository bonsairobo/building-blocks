use crate::{ChunkIndexer, OctreeSet, SmallKeyHashMap};

use building_blocks_core::prelude::*;

use serde::{Deserialize, Serialize};

/// A hash map of `OctreeSet`s which, unlike the vanilla `OctreeSet`, supports an unbounded set of points (within the bounds of
/// `Point3i`).
#[derive(Clone, Deserialize, Serialize)]
pub struct ChunkedOctreeSet {
    /// Indexer used to find the octree for a given chunk.
    pub indexer: ChunkIndexer<[i32; 3]>,

    octrees: SmallKeyHashMap<Point3i, OctreeSet>,
}

impl ChunkedOctreeSet {
    pub fn new(chunk_shape: Point3i, octrees: SmallKeyHashMap<Point3i, OctreeSet>) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            octrees,
        }
    }

    pub fn empty(chunk_shape: Point3i) -> Self {
        Self::new(chunk_shape, SmallKeyHashMap::default())
    }

    pub fn visit_octrees(&self, extent: &Extent3i, visitor: &mut impl FnMut(&OctreeSet)) {
        for chunk_min in self.indexer.chunk_mins_for_extent(extent) {
            if let Some(octree) = self.octrees.get(&chunk_min) {
                (visitor)(octree);
            }
        }
    }

    pub fn add_extent(&mut self, extent: &Extent3i) {
        let Self {
            octrees, indexer, ..
        } = self;

        for chunk_min in indexer.chunk_mins_for_extent(extent) {
            let octree = octrees.entry(chunk_min).or_insert_with(|| {
                let domain = Extent3i::from_min_and_shape(chunk_min, indexer.chunk_shape());

                OctreeSet::new_empty(domain)
            });
            octree.add_extent(extent);
        }
    }

    pub fn subtract_extent(&mut self, extent: &Extent3i) {
        let Self {
            octrees, indexer, ..
        } = self;

        let mut remove_chunks = Vec::new();
        for chunk_min in indexer.chunk_mins_for_extent(extent) {
            if let Some(octree) = octrees.get_mut(&chunk_min) {
                octree.subtract_extent(extent);
                if octree.is_empty() {
                    remove_chunks.push(chunk_min);
                }
            }
        }
        for chunk_min in remove_chunks.into_iter() {
            octrees.remove(&chunk_min);
        }
    }
}
