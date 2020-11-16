//! Algorithms for generating triangle meshes from:
//!   - height maps
//!   - signed distance fields
//!   - voxel occupancy grids
//!
//! All of the algorithms are designed to be used with a `ChunkMap`, such that each chunk will have
//! its own mesh. In order to update the mesh for a chunk, you must copy not only the chunk, but
//! also some adjacent points, into an array before running the meshing algorithm.
//!
//! An example of updating chunk meshes for a height map is shown below. The same general pattern
//! applies to all meshing algorithms, where you:
//!
//!   1. get the desired chunk extent
//!   2. pad the extent for a particular meshing algorithm
//!   3. copy that extent into an array
//!   4. mesh that array
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//! use building_blocks_mesh::height_map::*;
//!
//! use std::collections::HashSet;
//!
//! let chunk_shape = PointN([16; 2]);
//! let mut map = ChunkMap2::new(chunk_shape, 0.0, (), FastLz4 { level: 10 });
//!
//! // Mutate one or more of the chunks...
//! let mutated_chunk_keys = [PointN([0; 2]), PointN([16; 2])];
//!
//! // For each mutated chunk, and any adjacent chunk, the mesh will need to be updated.
//! let mut chunk_keys_to_update: HashSet<Point2i> = HashSet::new();
//! let offsets = Point2i::moore_offsets();
//! for chunk_key in mutated_chunk_keys.into_iter() {
//!     chunk_keys_to_update.insert(*chunk_key);
//!     for offset in offsets.iter() {
//!         chunk_keys_to_update.insert(*chunk_key + *offset * chunk_shape);
//!     }
//! }
//!
//! // Now we generate mesh vertices for each chunk.
//! let local_cache = LocalChunkCache2::new();
//! let reader = ChunkMapReader2::new(&map, &local_cache);
//! for chunk_key in chunk_keys_to_update.into_iter() {
//!     // It's crucial that we pad the chunk so we have access to adjacent points during meshing.
//!     let padded_chunk_extent = padded_height_map_chunk_extent(
//!         &map.extent_for_chunk_at_key(&chunk_key)
//!     );
//!     let mut padded_chunk = Array2::fill(padded_chunk_extent, 0.0);
//!     copy_extent(&padded_chunk_extent, &reader, &mut padded_chunk);
//!
//!     let mut hm_buffer = HeightMapMeshBuffer::default();
//!     triangulate_height_map(&padded_chunk, &padded_chunk_extent, &mut hm_buffer);
//!     // Do something with the mesh output...
//! }
//! ```
//!
//! All of the meshing algorithms are generic enough to work with an array wrapped in a
//! `TransformMap`.
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # use building_blocks_mesh::height_map::*;
//!
//! struct OtherHeight(f32);
//!
//! impl Height for OtherHeight {
//!     fn height(&self) -> f32 { self.0 }
//! }
//!
//! let extent = Extent2i::from_min_and_shape(PointN([0; 2]), PointN([50; 2]));
//! let array = Array2::fill(extent, 0.0);
//! let tfm_array = TransformMap::new(&array, |h: f32| OtherHeight(h));
//! let mut hm_buffer = HeightMapMeshBuffer::default();
//! triangulate_height_map(&tfm_array, &extent, &mut hm_buffer);
//! ```

pub mod greedy_quads;
pub mod height_map;
pub mod quad;
pub mod surface_nets;

pub use greedy_quads::{
    greedy_quads, padded_greedy_quads_chunk_extent, GreedyQuadsBuffer, MaterialVoxel,
};
pub use height_map::{
    padded_height_map_chunk_extent, triangulate_height_map, Height, HeightMapMeshBuffer,
};
pub use quad::{OrientedCubeFace, Quad, QuadGroup};
pub use surface_nets::{
    padded_surface_nets_chunk_extent, surface_nets, SignedDistance, SurfaceNetsBuffer,
};

#[derive(Clone, Default)]
pub struct PosNormMesh {
    pub positions: Vec<[f32; 3]>,
    /// Surface normal vectors. Not guaranteed to be normalized.
    pub normals: Vec<[f32; 3]>,
    /// All of the triangles in the mesh, wound counter-clockwise (right-hand rule).
    pub indices: Vec<u32>,
}

impl PosNormMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.indices.clear();
    }
}

#[derive(Clone, Default)]
pub struct PosNormTexMesh {
    pub positions: Vec<[f32; 3]>,
    /// Surface normal vectors. Not guaranteed to be normalized.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates, AKA UVs.
    pub tex_coords: Vec<[f32; 2]>,
    /// All of the triangles in the mesh, wound counter-clockwise (right-hand rule).
    pub indices: Vec<u32>,
}

impl PosNormTexMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.tex_coords.clear();
        self.indices.clear();
    }
}
