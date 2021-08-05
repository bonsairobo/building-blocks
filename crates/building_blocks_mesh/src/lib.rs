#![allow(
    clippy::type_complexity,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    nonstandard_style,
    unused,
    future_incompatible
)]
#![warn(clippy::doc_markdown)]
#![doc = include_str!("crate_doc.md")]

mod greedy_quads;
mod height_map;
mod quad;
mod surface_nets;

pub use greedy_quads::*;
pub use height_map::*;
pub use quad::*;
pub use surface_nets::*;

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

pub trait IsOpaque {
    /// Returns `true` if light cannot pass through this voxel.
    fn is_opaque(&self) -> bool;
}
