//! Structures and algorithms for implementing level of detail (LoD).
//!
//! More specifically, this module is called "multiresolution" because it helps with storing voxels of different "sizes" or
//! equivalently different sampling rates. Here's the breakdown of important structures we can use:
//!
//! - [OctreeSet](crate::OctreeSet): a bounded 3D hierarchical bitset, used for spatial indexing tasks
//! - [OctreeChunkIndex](crate::OctreeChunkIndex): an unbounded `OctreeSet` specifically for tracking the presence of chunks
//! - [ChunkPyramid](self::ChunkPyramid): an ordered list of `ChunkMap` "levels", where each level decreases the sampling rate
//! - [ChunkDownsampler](self::ChunkDownsampler): an algorithm for downsampling one chunk
//!
//! You will generally want to have a `ChunkPyramid` and a corresponding `OctreeChunkIndex` that tracks the set of chunks that
//! exist. Each node in the `OctreeChunkIndex` corresponds to a chunk at a particular level, i.e. an `ChunkKey`. There is
//! currently no enforcement of occupancy in the `ChunkPyramid`.
//!
//! `OctreeChunkIndex` is "unbounded" because it is actually a collection of `OctreeSet`s stored in a map. Each entry of that
//! map is called a "super chunk." You can think if it like a `ChunkMap`, except instead of `Array`s, it stores `OctreeSet`s.
//! Every superchunk is the same shape, and each is resonsible for a sparse set of chunks in a bounded region.
//!
//! You might wonder why the `OctreeChunkIndex` is necessary at all. It's main utility is for optimizing iteration over large
//! regions of the map. Without one, the best you could do is hash every single chunk key that overlaps your query extent to see
//! if it exists in the pyramid. It is also a natural structure for implementing a clipmap.
//!
//! ## Indexing and Downsampling a `ChunkMap`
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # use std::collections::HashSet;
//! #
//! // Constructing a pyramid is much the same as constructing a chunk map, except you need to supply a closure to construct
//! // empty storages.
//! let num_lods = 5; // Up to 6 supported for now.
//! let chunk_shape = Point3i::fill(16);
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder3x1::new(chunk_shape, ambient_value);
//! let mut pyramid = ChunkHashMapPyramid3::new(builder, || SmallKeyHashMap::new(), num_lods);
//!
//! // Populate LOD0, the highest resolution.
//! let mut lod0 = pyramid.level_mut(0);
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! lod0.fill_extent(&extent, 1);
//!
//! // Now we index the currently populated set of chunks.
//! let superchunk_shape = Point3i::fill(512);
//! let mut index = OctreeChunkIndex::index_chunk_map(superchunk_shape, lod0);
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
//! assert_eq!(chunk_keys, lod0.storage().chunk_keys().cloned().collect());
//!
//! // Now let's downsample those chunks into every level of the pyramid. This goes bottom-up in post-order. The
//! // `PointDownsampler` simply takes one point for each 2x2x2 region being sampled. There is also an `SdfMeanDownsampler` that
//! // works on any voxels types that implement `SignedDistance`. Or you can define your own downsampler!
//! pyramid.downsample_chunks_with_index(&index, &PointDownsampler, &extent);
//! ```

pub mod chunk_pyramid;
pub mod clipmap;
pub mod sampling;

pub use chunk_pyramid::*;
pub use clipmap::*;
pub use sampling::*;
