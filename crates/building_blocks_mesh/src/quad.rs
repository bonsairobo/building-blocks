use super::{PosNormMesh, PosNormTexMesh};

use building_blocks_core::{
    axis::{Axis3Permutation, SignedAxis3},
    prelude::*,
};

/// Metadata that's used to aid in the geometric calculations for one of the 6 possible cube faces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrientedCubeFace {
    /// Determines the orientation of the plane.
    pub n_sign: i32,

    /// Determines the {N, U, V} <--> {X, Y, Z} relation.
    pub permutation: Axis3Permutation,

    /// First in the `permutation` of +X, +Y, and +Z.
    pub n: Point3i,
    /// Second in the `permutation` of +X, +Y, and +Z.
    pub u: Point3i,
    /// Third in the `permutation` of +X, +Y, and +Z.
    pub v: Point3i,
}

impl OrientedCubeFace {
    pub fn new(n_sign: i32, permutation: Axis3Permutation) -> Self {
        let [n_axis, u_axis, v_axis] = permutation.axes();

        Self {
            n_sign,
            permutation,
            n: n_axis.get_unit_vector(),
            u: u_axis.get_unit_vector(),
            v: v_axis.get_unit_vector(),
        }
    }

    /// A cube face, using axes with an even permutation.
    pub fn canonical(normal: SignedAxis3) -> Self {
        Self::new(
            normal.sign,
            Axis3Permutation::even_with_normal_axis(normal.axis),
        )
    }

    pub fn quad_from_extent(&self, extent: &Extent3i) -> UnorientedQuad {
        UnorientedQuad {
            minimum: extent.minimum,
            width: self.u.dot(&extent.shape),
            height: self.v.dot(&extent.shape),
        }
    }

    pub fn quad_from_corners(&self, corner1: Point3i, corner2: Point3i) -> UnorientedQuad {
        self.quad_from_extent(&Extent3i::from_corners(corner1, corner2))
    }

    pub fn signed_normal(&self) -> Point3i {
        self.n * self.n_sign
    }

    pub fn mesh_normal(&self) -> Point3f {
        self.signed_normal().into()
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
    pub fn quad_corners(&self, quad: &UnorientedQuad) -> [Point3i; 4] {
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

        [minu_minv, maxu_minv, minu_maxv, maxu_maxv]
    }

    pub fn quad_mesh_positions(&self, quad: &UnorientedQuad) -> [[f32; 3]; 4] {
        let [c0, c1, c2, c3] = self.quad_corners(quad);

        [
            Point3f::from(c0).0,
            Point3f::from(c1).0,
            Point3f::from(c2).0,
            Point3f::from(c3).0,
        ]
    }

    pub fn quad_mesh_normals(&self) -> [[f32; 3]; 4] {
        [self.mesh_normal().0; 4]
    }

    /// Returns the 6 vertex indices for the quad in order to make two triangles in a mesh. Winding
    /// order depends on both the sign of the surface normal and the permutation of the UVs.
    pub fn quad_mesh_indices(&self, start: u32) -> [u32; 6] {
        quad_indices(start, self.n_sign * self.permutation.sign() > 0)
    }

    /// Extends `mesh` with the given `quad` that belongs to this face.
    pub fn add_quad_to_pos_norm_mesh(&self, quad: &UnorientedQuad, mesh: &mut PosNormMesh) {
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
    pub fn add_quad_to_pos_norm_tex_mesh(&self, quad: &UnorientedQuad, mesh: &mut PosNormTexMesh) {
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

/// The minimum voxel and size of a quad, without an orientation. To get the actual corners of the quad, combine with an
/// `OrientedCubeFace`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnorientedQuad {
    /// The minimum voxel in the quad.
    pub minimum: Point3i,
    /// Width of the quad.
    pub width: i32,
    /// Height of the quad.
    pub height: i32,
}

impl UnorientedQuad {
    pub fn from_voxel(voxel_point: Point3i) -> Self {
        Self {
            minimum: voxel_point,
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
