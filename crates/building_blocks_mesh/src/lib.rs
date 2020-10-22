pub mod height_map;
pub mod surface_nets;

// TODO: greedy meshing of cubes

pub use height_map::{
    padded_height_map_chunk_extent, triangulate_height_map, Height, HeightMapMeshBuffer,
};
pub use surface_nets::{
    padded_surface_nets_chunk_extent, surface_nets, SignedDistance, SurfaceNetsBuffer,
};

#[derive(Default)]
pub struct PosNormMesh {
    pub positions: Vec<[f32; 3]>,
    /// Surface normal vectors. Not guaranteed to be normalized.
    pub normals: Vec<[f32; 3]>,
    /// All of the triangles in the mesh, wound counter-clockwise (right-hand rule).
    pub indices: Vec<usize>,
}

impl PosNormMesh {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.indices.clear();
    }
}
