use crate::{Array3, ChunkIndexer, ChunkMap3, GetMut, IterChunkKeys, OctreeSet};

use building_blocks_core::prelude::*;

use fnv::FnvHashMap;

#[derive(Clone)]
pub struct OctreeChunkIndex {
    octrees: FnvHashMap<Point3i, OctreeSet>,
    indexer: ChunkIndexer<[i32; 3]>,
}

impl OctreeChunkIndex {
    pub fn index_chunk_map<T, Meta, Store>(
        superchunk_shape: Point3i,
        chunk_map: &ChunkMap3<T, Meta, Store>,
    ) -> Self
    where
        Store: for<'r> IterChunkKeys<'r, [i32; 3]>,
    {
        let chunk_shape = chunk_map.indexer.chunk_shape();

        Self::index_chunks(
            superchunk_shape,
            chunk_shape,
            chunk_map.storage().chunk_keys(),
        )
    }

    pub fn index_chunks<'a>(
        superchunk_shape: Point3i,
        chunk_shape: Point3i,
        chunk_keys: impl Iterator<Item = &'a Point3i>,
    ) -> Self {
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);

        let superchunk_mask = !(superchunk_shape - Point3i::ONES);

        let mut bitsets = FnvHashMap::default();
        for &chunk_key in chunk_keys {
            let chunk_p = chunk_key >> chunk_log2;
            let lod_key = chunk_p & superchunk_mask;
            let bitset = bitsets.entry(lod_key).or_insert_with(|| {
                let lod_chunk_extent = Extent3i::from_min_and_shape(lod_key, superchunk_shape);

                Array3::fill(lod_chunk_extent, false)
            });
            *bitset.get_mut(chunk_p) = true;
        }

        // PERF: could be done in parallel
        let mut octrees = FnvHashMap::default();
        for (lod_chunk_key, array) in bitsets.into_iter() {
            let octree = OctreeSet::from_array3(&array, *array.extent());
            octrees.insert(lod_chunk_key, octree);
        }

        Self {
            octrees,
            indexer: ChunkIndexer::new(superchunk_shape),
        }
    }

    pub fn visit_octrees(&self, extent: &Extent3i, visitor: &mut impl FnMut(&OctreeSet)) {
        for superchunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(octree) = self.octrees.get(&superchunk_key) {
                (visitor)(octree);
            }
        }
    }
}
