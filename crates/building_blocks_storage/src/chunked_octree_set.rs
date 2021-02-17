use crate::{ChunkIndexer, OctreeSet};

use building_blocks_core::prelude::*;

use fnv::FnvHashMap;
use serde::{Deserialize, Serialize};

/// A hash map of `OctreeSet`s which, unlike the vanilla `OctreeSet`, supports an unbounded set of points (within the bounds of
/// `Point3i`).
#[derive(Clone, Deserialize, Serialize)]
pub struct ChunkedOctreeSet {
    /// Indexer used to find the octree for a given chunk.
    pub indexer: ChunkIndexer<[i32; 3]>,

    octrees: FnvHashMap<Point3i, OctreeSet>,
}

impl ChunkedOctreeSet {
    pub fn new(chunk_shape: Point3i, octrees: FnvHashMap<Point3i, OctreeSet>) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            octrees,
        }
    }

    pub fn empty(chunk_shape: Point3i) -> Self {
        Self::new(chunk_shape, FnvHashMap::default())
    }

    pub fn visit_octrees(&self, extent: &Extent3i, visitor: &mut impl FnMut(&OctreeSet)) {
        for superchunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(octree) = self.octrees.get(&superchunk_key) {
                (visitor)(octree);
            }
        }
    }

    pub fn add_extent(&mut self, extent: &Extent3i) {
        let Self {
            octrees, indexer, ..
        } = self;

        for superchunk_key in indexer.chunk_keys_for_extent(extent) {
            let octree = octrees.entry(superchunk_key).or_insert_with(|| {
                let domain = Extent3i::from_min_and_shape(superchunk_key, indexer.chunk_shape());

                OctreeSet::empty(domain)
            });
            octree.add_extent(extent);
        }
    }

    pub fn subtract_extent(&mut self, extent: &Extent3i) {
        let Self {
            octrees, indexer, ..
        } = self;

        let mut remove_chunks = Vec::new();
        for superchunk_key in indexer.chunk_keys_for_extent(extent) {
            if let Some(octree) = octrees.get_mut(&superchunk_key) {
                octree.subtract_extent(extent);
                if octree.is_empty() {
                    remove_chunks.push(superchunk_key);
                }
            }
        }
        for superchunk_key in remove_chunks.into_iter() {
            octrees.remove(&superchunk_key);
        }
    }
}
