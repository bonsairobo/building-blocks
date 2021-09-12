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

use std::convert::TryInto;

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

    pub fn append(&mut self, other: &mut Self) {
        let n: u32 = self.positions.len().try_into().unwrap();

        self.positions.append(&mut other.positions);
        self.normals.append(&mut other.normals);

        self.indices.extend(other.indices.drain(..).map(|i| n + i));
    }

    /// Create a new mesh with equivalent triangles such that no vertex is shared by any two triangles.
    ///
    /// Also computes a normal for each triangle using the cross product. The pre-existing normals are not used.
    pub fn process_for_flat_shading(&self) -> PosNormMesh {
        let indices_len = self.indices.len();
        let mut mesh = PosNormMesh {
            positions: Vec::with_capacity(indices_len),
            normals: Vec::with_capacity(indices_len),
            indices: Vec::new(),
        };

        for triangle_i in self.indices.chunks(3) {
            let p1 = self.positions[triangle_i[0] as usize];
            let p2 = self.positions[triangle_i[1] as usize];
            let p3 = self.positions[triangle_i[2] as usize];

            let u = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let v = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];

            mesh.positions.push(p1);
            mesh.positions.push(p2);
            mesh.positions.push(p3);

            mesh.normals.push(n);
            mesh.normals.push(n);
            mesh.normals.push(n);
        }

        mesh
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
