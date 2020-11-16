use super::{PosNormMesh, PosNormTexMesh};

use building_blocks_core::{
    axis::{Axis3Permutation, SignedAxis3},
    prelude::*,
};

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
        let [n_axis, u_axis, v_axis] = permutation.axes();
        let n = n_axis.get_unit_vector();
        let mesh_normal: Point3f = (n * n_sign).into();

        Self {
            n_sign,

            permutation,

            n,
            u: u_axis.get_unit_vector(),
            v: v_axis.get_unit_vector(),

            mesh_normal,
        }
    }

    pub fn canonical(normal: SignedAxis3) -> Self {
        Self::new(
            normal.sign,
            Axis3Permutation::even_with_normal_axis(normal.axis),
        )
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

    pub fn quad_mesh_positions(&self, quad: &Quad) -> [[f32; 3]; 4] {
        let [c0, c1, c2, c3] = self.quad_corners(quad);

        [c0.0, c1.0, c2.0, c3.0]
    }

    pub fn quad_mesh_normals(&self) -> [[f32; 3]; 4] {
        [self.mesh_normal.0; 4]
    }

    /// Returns the 6 vertex indices for the quad in order to make two triangles in a mesh. Winding
    /// order depends on both the sign of the surface normal and the permutation of the UVs.
    pub fn quad_mesh_indices(&self, start: u32) -> [u32; 6] {
        quad_indices(start, self.n_sign * self.permutation.sign() > 0)
    }

    /// Extends `mesh` with the given `quad` that belongs to this face.
    pub fn add_quad_to_pos_norm_mesh(&self, quad: &Quad, mesh: &mut PosNormMesh) {
        let start_index = mesh.positions.len() as u32;
        mesh.positions
            .extend_from_slice(&self.quad_mesh_positions(quad));
        mesh.normals.extend_from_slice(&self.quad_mesh_normals());
        mesh.indices
            .extend_from_slice(&self.quad_mesh_indices(start_index));
    }

    /// Extends `mesh` with the given `quad` that belongs to this group.
    ///
    /// The texture coordinates come from `Quad::simple_tex_coords`.
    pub fn add_quad_to_pos_norm_tex_mesh(&self, quad: &Quad, mesh: &mut PosNormTexMesh) {
        let start_index = mesh.positions.len() as u32;
        mesh.positions
            .extend_from_slice(&self.quad_mesh_positions(quad));
        mesh.normals.extend_from_slice(&self.quad_mesh_normals());
        mesh.tex_coords.extend_from_slice(&quad.simple_tex_coords());
        mesh.indices
            .extend_from_slice(&self.quad_mesh_indices(start_index));
    }
}

/// Returns the vertex indices for a single quad (two triangles). The triangles may have either
/// clockwise or counter-clockwise winding. `start` is the first index.
pub fn quad_indices(start: u32, counter_clockwise: bool) -> [u32; 6] {
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
    pub fn for_voxel_face(voxel_point: Point3i, face: SignedAxis3) -> Self {
        let mut minimum = voxel_point;
        if face.sign > 0 {
            minimum += face.axis.get_unit_vector();
        }

        Self {
            minimum,
            width: 1,
            height: 1,
        }
    }

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
