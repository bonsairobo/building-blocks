pub mod greedy_quads;
pub mod height_map;
pub mod surface_nets;

pub use greedy_quads::{
    greedy_quads, padded_greedy_quads_chunk_extent, pos_norm_tex_meshes_from_material_quads,
    GreedyQuadsBuffer, MaterialVoxel, Quad,
};
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

#[derive(Default)]
pub struct PosNormTexMesh {
    pub positions: Vec<[f32; 3]>,
    /// Surface normal vectors. Not guaranteed to be normalized.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates, AKA UVs.
    pub tex_coords: Vec<[f32; 2]>,
    /// All of the triangles in the mesh, wound counter-clockwise (right-hand rule).
    pub indices: Vec<usize>,
}

impl PosNormTexMesh {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.tex_coords.clear();
        self.indices.clear();
    }
}
