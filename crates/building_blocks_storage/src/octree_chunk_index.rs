use crate::{
    active_clipmap_lod_chunks, Array3, ChunkIndexer, ChunkMap3, ClipMapConfig3, ClipMapUpdate3,
    GetMut, IterChunkKeys, LodChunkKey3, LodChunkUpdate3, OctreeSet,
};

use building_blocks_core::prelude::*;

use fnv::FnvHashMap;

#[derive(Clone)]
pub struct OctreeChunkIndex {
    /// Indexer used to find the octree for a given superchunk.
    pub indexer: ChunkIndexer<[i32; 3]>,

    chunk_shape: Point3i,
    octrees: FnvHashMap<Point3i, OctreeSet>,
}

impl OctreeChunkIndex {
    /// The shape of the world extent convered by a single chunk (a leaf of an octree).
    pub fn chunk_shape(&self) -> Point3i {
        self.chunk_shape
    }

    /// The shape of the world extent covered by a single octree, i.e. all of its chunks when full.
    pub fn superchunk_shape(&self) -> Point3i {
        self.indexer.chunk_shape()
    }

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
        assert!(superchunk_shape.dimensions_are_powers_of_2());
        assert!(chunk_shape.dimensions_are_powers_of_2());

        let superchunk_log2 = superchunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        assert!(superchunk_log2 > chunk_log2);

        let superchunk_mask = !((superchunk_shape >> chunk_log2) - Point3i::ONES);

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
            chunk_shape,
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

    pub fn clipmap_config(&self, clip_box_radius: i32) -> ClipMapConfig3 {
        assert!(self.indexer.chunk_shape().is_cube());
        assert!(self.chunk_shape().is_cube());

        let superchunk_log2 = self.indexer.chunk_shape().x().trailing_zeros() as u8;
        let chunk_log2 = self.chunk_shape().x().trailing_zeros() as u8;
        let num_lods = superchunk_log2 - chunk_log2;

        ClipMapConfig3 {
            num_lods,
            chunk_shape: self.chunk_shape(),
            clip_box_radius,
        }
    }

    pub fn active_clipmap_lod_chunks(
        &self,
        extent: &Extent3i,
        clip_box_radius: i32,
        lod0_center: Point3i,
        mut init_rx: impl FnMut(LodChunkKey3),
    ) {
        let config = self.clipmap_config(clip_box_radius);
        self.visit_octrees(extent, &mut |octree| {
            active_clipmap_lod_chunks(&config, octree, lod0_center, &mut init_rx)
        });
    }

    pub fn find_clipmap_chunk_updates(
        &self,
        extent: &Extent3i,
        clip_box_radius: i32,
        old_lod0_center: Point3i,
        new_lod0_center: Point3i,
        mut update_rx: impl FnMut(LodChunkUpdate3),
    ) {
        let update = ClipMapUpdate3::new(
            &self.clipmap_config(clip_box_radius),
            old_lod0_center,
            new_lod0_center,
        );
        self.visit_octrees(extent, &mut |octree| {
            update.find_chunk_updates(octree, &mut update_rx)
        });
    }
}
