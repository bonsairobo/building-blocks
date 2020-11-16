use super::{PosNormMesh, PosNormTexMesh};

use building_blocks_core::{axis::Axis3Permutation, prelude::*};

/// A set of `Quad`s that share an orientation.
pub struct QuadGroup<M> {
    /// The quads themselves. We rely on the group's metadata to interpret them.
    pub quads: Vec<(Quad, M)>,
    pub face: OrientedCubeFace,
}

impl<M> QuadGroup<M> {
    pub fn new(face: OrientedCubeFace) -> Self {
        Self {
            quads: Vec::new(),
            face,
        }
    }
}

/// Metadata that's used to aid in the geometric calculations for one of the 6 possible cube faces.
pub struct OrientedCubeFace {
    // Determines the orientation of the plane.
    pub n_sign: i32,

    // Determines the {N, U, V} <--> {X, Y, Z} relation.
    pub permutation: Axis3Permutation,

    // These vectors are always some permutation of +X, +Y, and +Z.
    pub n: Point3i,
    pub u: Point3i,
    pub v: Point3i,

    pub mesh_normal: Point3f,
}

impl OrientedCubeFace {
    pub fn new(n_sign: i32, permutation: Axis3Permutation) -> Self {
        let xyz = [PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])];

        let [n_axis, u_axis, v_axis] = permutation.axes();
        let n = xyz[n_axis.index()];
        let mesh_normal: Point3f = (n * n_sign).into();

        Self {
            n_sign,

            permutation,

            n,
            u: xyz[u_axis.index()],
            v: xyz[v_axis.index()],

            mesh_normal,
        }
    }

    /// Returns the 4 corners of the quad in this order:
    ///
    /// ```text
    ///         2 ----> 3
    ///           ^
    ///     ^       \
    ///     |         \
    ///  +v |   0 ----> 1
    ///     |
    ///      -------->
    ///        +u
    /// ```
    pub fn quad_corners(&self, quad: &Quad) -> [Point3f; 4] {
        let w_vec = self.u * quad.width;
        let h_vec = self.v * quad.height;

        let minu_minv = if self.n_sign > 0 {
            quad.minimum + self.n
        } else {
            quad.minimum
        };
        let maxu_minv = minu_minv + w_vec;
        let minu_maxv = minu_minv + h_vec;
        let maxu_maxv = minu_minv + w_vec + h_vec;

        [
            minu_minv.into(),
            maxu_minv.into(),
            minu_maxv.into(),
            maxu_maxv.into(),
        ]
    }

    /// Extends `mesh` with the given `quad` that belongs to this group.
    pub fn add_quad_to_pos_norm_mesh(&self, quad: &Quad, mesh: &mut PosNormMesh) {
        let cur_idx = mesh.positions.len();
        mesh.indices.extend_from_slice(&self.indices(cur_idx));
        let [c0, c1, c2, c3] = self.quad_corners(quad);
        mesh.positions.extend_from_slice(&[c0.0, c1.0, c2.0, c3.0]);
        mesh.normals.extend_from_slice(&[self.mesh_normal.0; 4]);
    }

    /// Extends `mesh` with the given `quad` that belongs to this group.
    ///
    /// The texture coordinates come from `Quad::simple_tex_coords`.
    pub fn add_quad_to_pos_norm_tex_mesh(&self, quad: &Quad, mesh: &mut PosNormTexMesh) {
        let cur_idx = mesh.positions.len();
        mesh.indices.extend_from_slice(&self.indices(cur_idx));
        let [c0, c1, c2, c3] = self.quad_corners(quad);
        mesh.positions.extend_from_slice(&[c0.0, c1.0, c2.0, c3.0]);
        mesh.normals.extend_from_slice(&[self.mesh_normal.0; 4]);
        mesh.tex_coords.extend_from_slice(&quad.simple_tex_coords());
    }

    /// Returns the 6 vertex indices for the quad in order to make two triangles in a mesh.
    pub fn indices(&self, start: usize) -> [usize; 6] {
        quad_indices(start, self.n_sign * self.permutation.sign() > 0)
    }
}

/// Returns the vertex indices for a single quad (two triangles). The triangles may have either
/// clockwise or counter-clockwise winding. `start` is the first index.
pub fn quad_indices(start: usize, counter_clockwise: bool) -> [usize; 6] {
    if counter_clockwise {
        [start, start + 1, start + 2, start + 1, start + 3, start + 2]
    } else {
        [start, start + 2, start + 1, start + 1, start + 2, start + 3]
    }
}

/// A single quad of connected cubic voxel faces. Must belong to a `QuadGroup` to be useful.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quad {
    pub minimum: Point3i,
    pub width: i32,
    pub height: i32,
}

impl Quad {
    /// Returns the UV coordinates of the 4 corners of the quad. Returns in the same order as
    /// `OrientedCubeFace::quad_corners`.
    ///
    /// This is just one way of assigning UVs to voxel quads. It assumes that each material has a
    /// single tile texture with wrapping coordinates, and each voxel face should show the entire
    /// texture. It also assumes a particular orientation for the texture. This should be sufficient
    /// for minecraft-style meshing.
    ///
    /// If you need to use a texture atlas, you must calculate your own coordinates from the `Quad`.
    pub fn simple_tex_coords(&self) -> [[f32; 2]; 4] {
        [
            [0.0, 0.0],
            [self.width as f32, 0.0],
            [0.0, self.height as f32],
            [self.width as f32, self.height as f32],
        ]
    }
}
