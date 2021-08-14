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
//! let superchunk_exponent = 9;
//! let chunk_exponent = 4;
//! let chunk_shape = Point3i::fill(1 << chunk_exponent);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(ChunkMapConfig { chunk_shape, ambient_value, root_lod: 0 });
//! let mut map = builder.build_with_hash_map_storage();
//!
//! // Populate LOD0, the highest resolution.
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! map.fill_extent(0, &extent, 1);
//!
//! // Now we index the currently populated set of chunks.
//! let num_lods = 5; // Up to 6 supported for now.
//! let mut index = OctreeChunkIndex::index_chunk_map(superchunk_exponent, num_lods, &map);
//!
//! // Just make sure everything's here. A unit test to help you understand this structure.
//! let mut chunk_keys = HashSet::new();
//! index.visit_octrees(
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
//! map.downsample_chunks_with_index(&index, &PointDownsampler, &extent);
//! ```

use crate::{
    dev_prelude::{
        Array3x1, ChunkKey3, ChunkMap3, ChunkUnits, ChunkedOctreeSet, ClipMapConfig3,
        ClipMapUpdate3, GetMutUnchecked, IterChunkKeys, LodChunkUpdate3, OctreeSet,
        SmallKeyHashMap,
    },
    octree::active_clipmap_lod_chunks,
};

use building_blocks_core::prelude::*;

use serde::{Deserialize, Serialize};

/// A `ChunkedOctreeSet` that indexes the chunks of a `ChunkMap`. Useful for representing a clipmap.
#[derive(Clone, Deserialize, Serialize)]
pub struct OctreeChunkIndex {
    /// An unbounded set of chunk keys, but scaled down to be contiguous, i.e. it operates in `ChunkUnits`.
    superchunk_octrees: ChunkedOctreeSet,

    superchunk_exponent: u8,
    chunk_exponent: u8,
    /// Independent from the superchunk_exponent, although it may not be larger than self.max_lods().
    num_lods: u8,
}

impl OctreeChunkIndex {
    pub fn new_empty(superchunk_exponent: u8, chunk_exponent: u8, num_lods: u8) -> Self {
        validate_params(superchunk_exponent, chunk_exponent, num_lods);

        Self {
            superchunk_octrees: ChunkedOctreeSet::new_empty(Point3i::fill(
                1i32 << superchunk_exponent,
            )),
            superchunk_exponent,
            chunk_exponent,
            num_lods,
        }
    }

    pub fn superchunk_exponent(&self) -> u8 {
        self.superchunk_exponent
    }

    pub fn chunk_exponent(&self) -> u8 {
        self.chunk_exponent
    }

    #[inline]
    pub fn num_lods(&self) -> u8 {
        self.num_lods
    }

    /// The shape of the world extent convered by a single chunk (a leaf of an octree).
    #[inline]
    pub fn chunk_shape(&self) -> Point3i {
        Point3i::fill(1i32 << self.chunk_exponent)
    }

    /// The shape of the world extent covered by a single octree, i.e. all of its chunks when full.
    #[inline]
    pub fn superchunk_shape(&self) -> Point3i {
        self.superchunk_octrees.indexer.chunk_shape()
    }

    /// Same as `index_lod0_chunks`, but using the chunk keys and chunk shape from `chunk_map`.
    pub fn index_chunk_map<T, Ch, Store>(
        superchunk_exponent: u8,
        num_lods: u8,
        chunk_map: &ChunkMap3<T, Ch, Store>,
    ) -> Self
    where
        Store: for<'r> IterChunkKeys<'r, [i32; 3]>,
    {
        assert!(chunk_map.indexer.chunk_shape().is_cube());
        let chunk_exponent = chunk_map.indexer.chunk_shape().x().trailing_zeros() as u8;

        Self::index_lod0_chunks(
            superchunk_exponent,
            chunk_exponent,
            num_lods,
            chunk_map.storage().chunk_keys().filter(|k| k.lod == 0),
        )
    }

    /// Create a new `OctreeChunkIndex` whose octrees contain exactly the set `chunk_keys`. The number of levels in an octree
    /// corresponds to the relative sizes of the chunks and superchunks. A superchunk is a chunk of the domain that contains a
    /// single octree of many smaller chunks. Superchunk shape, like chunk shape, must have all dimensions be powers of 2.
    /// Because of the static limitations on `OctreeSet` size, you can only have up to 6 levels of detail. This means
    /// `superchunk_exponent - chunk_exponent` must be less than `6`. For example, if your chunk shape is `[2^4, 2^4, 2^4]`,
    /// then your superchunk shape can be at most `[2^9, 2^9, 2^9]`.
    pub fn index_lod0_chunks<'a>(
        superchunk_exponent: u8,
        chunk_exponent: u8,
        num_lods: u8,
        chunk_keys: impl Iterator<Item = &'a ChunkKey3>,
    ) -> Self {
        validate_params(superchunk_exponent, chunk_exponent, num_lods);

        let superchunk_exponent_in_chunks =
            Point3i::fill(1i32 << (superchunk_exponent - chunk_exponent));

        let superchunk_mask = Point3i::fill(!((1i32 << superchunk_exponent) - 1));

        let mut superchunk_bitsets = SmallKeyHashMap::default();
        for chunk_key in chunk_keys {
            assert_eq!(chunk_key.lod, 0);
            let superchunk_min = chunk_key.minimum & superchunk_mask;
            let bitset = superchunk_bitsets.entry(superchunk_min).or_insert_with(|| {
                Array3x1::fill(
                    Extent3i::from_min_and_shape(
                        superchunk_min >> chunk_exponent,
                        superchunk_exponent_in_chunks,
                    ),
                    false,
                )
            });
            unsafe {
                *bitset.get_mut_unchecked(chunk_key.minimum >> chunk_exponent) = true;
            }
        }

        // PERF: could be done in parallel
        let mut superchunk_octrees = SmallKeyHashMap::default();
        for (lod_chunk_key, bitset) in superchunk_bitsets.into_iter() {
            let octree = OctreeSet::from_array3(&bitset, *bitset.extent());
            superchunk_octrees.insert(lod_chunk_key, octree);
        }

        Self {
            superchunk_octrees: ChunkedOctreeSet::new(
                Point3i::fill(1i32 << superchunk_exponent),
                superchunk_octrees,
            ),
            superchunk_exponent,
            chunk_exponent,
            num_lods,
        }
    }

    /// Inserts all of the chunks for one superchunk. Panics if any of the chunk keys fall outside of the superchunk extent.
    /// Returns the old `OctreeSet` for the superchunk if one existed.
    pub fn insert_superchunk(
        &mut self,
        superchunk_min: Point3i,
        chunk_keys: impl Iterator<Item = ChunkKey3>,
    ) -> Option<OctreeSet> {
        let superchunk_extent =
            Extent3i::from_min_and_shape(superchunk_min, self.superchunk_shape());
        let super_chunk_extent_in_chunks = superchunk_extent >> self.chunk_exponent;

        let mut bitset = Array3x1::fill(super_chunk_extent_in_chunks, false);
        for chunk_key in chunk_keys {
            assert_eq!(chunk_key.lod, 0);
            unsafe {
                *bitset.get_mut_unchecked(chunk_key.minimum >> self.chunk_exponent) = true;
            }
        }
        let octree = OctreeSet::from_array3(&bitset, *bitset.extent());

        self.superchunk_octrees.insert_chunk(superchunk_min, octree)
    }

    pub fn pop_superchunk(&mut self, superchunk_min: Point3i) -> Option<OctreeSet> {
        self.superchunk_octrees.pop_chunk(superchunk_min)
    }

    pub fn clipmap_config(&self, clip_box_radius: ChunkUnits<u16>) -> ClipMapConfig3 {
        assert!(self.superchunk_octrees.indexer.chunk_shape().is_cube());
        assert!(self.chunk_shape().is_cube());

        ClipMapConfig3::new(self.num_lods, clip_box_radius, self.chunk_shape())
    }

    /// Traverses all octree nodes overlapping `extent` to find the `ChunkKey3`s that are "active" when the clipmap is
    /// centered at `lod0_center`.
    pub fn active_clipmap_lod_chunks(
        &self,
        extent: &Extent3i,
        clip_box_radius: ChunkUnits<u16>,
        lod0_center: ChunkUnits<Point3i>,
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
        clip_box_radius: ChunkUnits<u16>,
        old_lod0_center: ChunkUnits<Point3i>,
        new_lod0_center: ChunkUnits<Point3i>,
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

    pub fn add_extent(&mut self, extent: ChunkUnits<Extent3i>) {
        self.superchunk_octrees.add_extent(&extent.0)
    }

    pub fn subtract_extent(&mut self, extent: ChunkUnits<Extent3i>) {
        self.superchunk_octrees.subtract_extent(&extent.0)
    }

    /// Visit all superchunk octrees that overlap `extent`.
    pub fn visit_octrees(&self, extent: &Extent3i, visitor: &mut impl FnMut(&OctreeSet)) {
        self.superchunk_octrees.visit_octrees(extent, visitor)
    }
}

fn validate_params(superchunk_exponent: u8, chunk_exponent: u8, num_lods: u8) {
    assert!(superchunk_exponent > chunk_exponent);
    assert!(
        superchunk_exponent - chunk_exponent < 6,
        "OctreeSet only support 6 levels. Make your chunk shape larger or make your superchunk shape smaller.
            superchunk shape = {:?}, log2 = {:?}
            chunk shape      = {:?}, log2 = {:?}",
        superchunk_exponent,
        superchunk_exponent,
        chunk_exponent,
        chunk_exponent
    );

    let max_lods = superchunk_exponent - chunk_exponent + 1;
    assert!(num_lods <= max_lods);
}
