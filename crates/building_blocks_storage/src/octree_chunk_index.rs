use crate::{
    active_clipmap_lod_chunks, Array3, ChunkMap3, ChunkedOctreeSet, ClipMapConfig3, ClipMapUpdate3,
    GetMut, IterChunkKeys, LodChunkKey3, LodChunkUpdate3, OctreeSet,
};

use building_blocks_core::prelude::*;

use fnv::FnvHashMap;
use serde::{Deserialize, Serialize};

/// A `ChunkedOctreeSet` that indexes the chunks of a `ChunkMap` or a `ChunkPyramid`. Useful for representing a clipmap.
#[derive(Clone, Deserialize, Serialize)]
pub struct OctreeChunkIndex {
    /// An unbounded set of chunk keys, but scaled down to be contiguous. For example, if the chunk shape is `16^3`, then the
    /// chunk key `[16, 32, -64]` is represented as point `[1, 2, -4]` in this set.
    pub superchunk_octrees: ChunkedOctreeSet,
    chunk_shape: Point3i,
}

impl OctreeChunkIndex {
    /// The shape of the world extent convered by a single chunk (a leaf of an octree).
    #[inline]
    pub fn chunk_shape(&self) -> Point3i {
        self.chunk_shape
    }

    /// The shape of the world extent covered by a single octree, i.e. all of its chunks when full.
    #[inline]
    pub fn superchunk_shape(&self) -> Point3i {
        self.superchunk_octrees.indexer.chunk_shape()
    }

    /// Same as `index_chunks`, but using the chunk keys and chunk shape from `chunk_map`.
    pub fn index_chunk_map<T, Ch, Store>(
        superchunk_shape: Point3i,
        chunk_map: &ChunkMap3<T, Ch, Store>,
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

    /// Create a new `OctreeChunkIndex` whose octrees contain exactly the set `chunk_keys`. The number of levels in an octree
    /// corresponds to the relative sizes of the chunks and superchunks. A superchunk is a chunk of the domain that contains a
    /// single octree of many smaller chunks. Superchunk shape, like chunk shape, must have all dimensions be powers of 2.
    /// Because of the static limitations on `OctreeSet` size, you can only have up to 6 levels of detail. This means
    /// `superchunk_shape / chunk_shape` must be less than `2 ^ [6, 6, 6] = [64, 64, 64]`. For example, if your chunk shape is
    /// `[16, 16, 16]`, then your superchunk shape can be at most `[512, 512, 512]`.
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

        assert!(
            superchunk_log2 - chunk_log2 < Point3i::fill(6),
            "OctreeSet only support 6 levels. Make your chunk shape larger or make your superchunk shape smaller"
        );

        let superchunk_shape_in_chunks = superchunk_shape >> chunk_log2;

        let superchunk_mask = !(superchunk_shape - Point3i::ONES);

        let mut superchunk_bitsets = FnvHashMap::default();
        for &chunk_key in chunk_keys {
            let superchunk_key = chunk_key & superchunk_mask;
            let bitset = superchunk_bitsets.entry(superchunk_key).or_insert_with(|| {
                Array3::fill(
                    Extent3i::from_min_and_shape(
                        superchunk_key >> chunk_log2,
                        superchunk_shape_in_chunks,
                    ),
                    false,
                )
            });
            *bitset.get_mut(chunk_key >> chunk_log2) = true;
        }

        // PERF: could be done in parallel
        let mut superchunk_octrees = FnvHashMap::default();
        for (lod_chunk_key, array) in superchunk_bitsets.into_iter() {
            let octree = OctreeSet::from_array3(&array, *array.extent());
            superchunk_octrees.insert(lod_chunk_key, octree);
        }

        Self {
            superchunk_octrees: ChunkedOctreeSet::new(superchunk_shape, superchunk_octrees),
            chunk_shape,
        }
    }

    pub fn clipmap_config(&self, clip_box_radius: i32) -> ClipMapConfig3 {
        assert!(self.superchunk_octrees.indexer.chunk_shape().is_cube());
        assert!(self.chunk_shape().is_cube());

        let superchunk_log2 = self
            .superchunk_octrees
            .indexer
            .chunk_shape()
            .x()
            .trailing_zeros() as u8;
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
        self.superchunk_octrees
            .visit_octrees(extent, &mut |octree| {
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
        self.superchunk_octrees
            .visit_octrees(extent, &mut |octree| {
                update.find_chunk_updates(octree, &mut update_rx)
            });
    }
}
