//! `OctreeChunkIndex` is "unbounded" because it is actually a collection of `OctreeSet`s stored in a map. Each entry of that
//! map is called a "super chunk." You can think if it like a `ChunkMap`, except instead of `Array`s, it stores `OctreeSet`s.
//! Every superchunk is the same shape, and each is resonsible for a sparse set of chunks in a bounded region.
//!
//! You might wonder why the `OctreeChunkIndex` is necessary at all. It's main utility is for optimizing iteration over large
//! regions of the map. Without one, the best you could do is hash every single `ChunkKey` that overlaps your query extent to
//! see if it exists in the `ChunkMap`. It is also a natural structure for implementing a clipmap.
//!
//! ## Indexing and Downsampling a `ChunkMap`
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # use std::collections::HashSet;
//! #
//! let chunk_shape = Point3i::fill(16);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(chunk_shape, ambient_value);
//! let mut map = builder.build_with_hash_map_storage();
//!
//! // Populate LOD0, the highest resolution.
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! map.fill_extent(0, &extent, 1);
//!
//! // Now we index the currently populated set of chunks.
//! let superchunk_shape = Point3i::fill(512);
//! let mut index = OctreeChunkIndex::index_chunk_map(superchunk_shape, &map);
//!
//! // Just make sure everything's here. A unit test to help you understand this structure.
//! let mut chunk_keys = HashSet::new();
//! index.superchunk_octrees.visit_octrees(
//!     &extent,
//!     &mut |octree: &OctreeSet| {
//!         octree.visit_all_octants_in_preorder(&mut |node: &OctreeNode| {
//!             // Chunks are the single-voxel leaves. Remember this octree is indexing in a space where 1 voxel = 1 chunk.
//!             if node.octant().is_single_voxel() {
//!                 // The octree coordinates are downscaled by the chunk shape.
//!                 chunk_keys.insert(ChunkKey::new(0, node.octant().minimum() * chunk_shape));
//!             }
//!             VisitStatus::Continue
//!         });
//!     }
//! );
//! assert_eq!(chunk_keys, map.storage().chunk_keys().cloned().collect());
//!
//! // Now let's downsample those chunks into every LOD of the map. This goes bottom-up in post-order. The
//! // `PointDownsampler` simply takes one point for each 2x2x2 region being sampled. There is also an `SdfMeanDownsampler` that
//! // works on any voxels types that implement `SignedDistance`. Or you can define your own downsampler!
//! let num_lods = 5; // Up to 6 supported for now.
//! map.downsample_chunks_with_index(num_lods, &index, &PointDownsampler, &extent);
//! ```

use crate::{
    active_clipmap_lod_chunks, Array3x1, ChunkKey3, ChunkMap3, ChunkUnits3, ChunkedOctreeSet,
    ClipMapConfig3, ClipMapUpdate3, GetMut, IterChunkKeys, LodChunkUpdate3, OctreeSet,
    SmallKeyHashMap,
};

use building_blocks_core::prelude::*;

use serde::{Deserialize, Serialize};

/// A `ChunkedOctreeSet` that indexes the chunks of a `ChunkMap`. Useful for representing a clipmap.
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

    /// Same as `index_lod0_chunks`, but using the chunk keys and chunk shape from `chunk_map`.
    pub fn index_chunk_map<T, Ch, Store>(
        superchunk_shape: Point3i,
        chunk_map: &ChunkMap3<T, Ch, Store>,
    ) -> Self
    where
        Store: for<'r> IterChunkKeys<'r, [i32; 3]>,
    {
        let chunk_shape = chunk_map.indexer.chunk_shape();

        Self::index_lod0_chunks(
            superchunk_shape,
            chunk_shape,
            chunk_map.storage().chunk_keys().filter(|k| k.lod == 0),
        )
    }

    /// Create a new `OctreeChunkIndex` whose octrees contain exactly the set `chunk_keys`. The number of levels in an octree
    /// corresponds to the relative sizes of the chunks and superchunks. A superchunk is a chunk of the domain that contains a
    /// single octree of many smaller chunks. Superchunk shape, like chunk shape, must have all dimensions be powers of 2.
    /// Because of the static limitations on `OctreeSet` size, you can only have up to 6 levels of detail. This means
    /// `superchunk_shape / chunk_shape` must be less than `2 ^ [6, 6, 6] = [64, 64, 64]`. For example, if your chunk shape is
    /// `[16, 16, 16]`, then your superchunk shape can be at most `[512, 512, 512]`.
    pub fn index_lod0_chunks<'a>(
        superchunk_shape: Point3i,
        chunk_shape: Point3i,
        chunk_keys: impl Iterator<Item = &'a ChunkKey3>,
    ) -> Self {
        assert!(superchunk_shape.dimensions_are_powers_of_2());
        assert!(chunk_shape.dimensions_are_powers_of_2());

        let superchunk_log2 = superchunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        assert!(superchunk_log2 > chunk_log2);

        assert!(
            superchunk_log2 - chunk_log2 < Point3i::fill(6),
            "OctreeSet only support 6 levels. Make your chunk shape larger or make your superchunk shape smaller.
             superchunk shape = {:?}, log2 = {:?}
             chunk shape      = {:?}, log2 = {:?}",
            superchunk_shape,
            superchunk_log2,
            chunk_shape,
            chunk_log2
        );

        let superchunk_shape_in_chunks = superchunk_shape >> chunk_log2;

        let superchunk_mask = !(superchunk_shape - Point3i::ONES);

        let mut superchunk_bitsets = SmallKeyHashMap::default();
        for chunk_key in chunk_keys {
            let superchunk_key = chunk_key.minimum & superchunk_mask;
            let bitset = superchunk_bitsets.entry(superchunk_key).or_insert_with(|| {
                Array3x1::fill(
                    Extent3i::from_min_and_shape(
                        superchunk_key >> chunk_log2,
                        superchunk_shape_in_chunks,
                    ),
                    false,
                )
            });
            *bitset.get_mut(chunk_key.minimum >> chunk_log2) = true;
        }

        // PERF: could be done in parallel
        let mut superchunk_octrees = SmallKeyHashMap::default();
        for (lod_chunk_key, bitset) in superchunk_bitsets.into_iter() {
            let octree = OctreeSet::from_array3(&bitset, *bitset.extent());
            superchunk_octrees.insert(lod_chunk_key, octree);
        }

        Self {
            superchunk_octrees: ChunkedOctreeSet::new(superchunk_shape, superchunk_octrees),
            chunk_shape,
        }
    }

    pub fn clipmap_config(&self, clip_box_radius: u16) -> ClipMapConfig3 {
        assert!(self.superchunk_octrees.indexer.chunk_shape().is_cube());
        assert!(self.chunk_shape().is_cube());

        // Calculations assume the chunk shape is a cube.
        let superchunk_log2 = self
            .superchunk_octrees
            .indexer
            .chunk_shape()
            .x()
            .trailing_zeros() as u8;
        let chunk_log2 = self.chunk_shape().x().trailing_zeros() as u8;
        let num_lods = superchunk_log2 - chunk_log2 + 1;

        ClipMapConfig3::new(num_lods, clip_box_radius, self.chunk_shape())
    }

    /// Traverses all octree nodes overlapping `extent` to find the `ChunkKey3`s that are "active" when the clipmap is
    /// centered at `lod0_center`.
    pub fn active_clipmap_lod_chunks(
        &self,
        extent: &Extent3i,
        clip_box_radius: u16,
        lod0_center: ChunkUnits3,
        mut init_rx: impl FnMut(ChunkKey3),
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
        clip_box_radius: u16,
        old_lod0_center: ChunkUnits3,
        new_lod0_center: ChunkUnits3,
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
