use super::{PosNormMesh, PosNormTexMesh};

use building_blocks_core::prelude::*;
use building_blocks_storage::{access::GetUncheckedRelease, prelude::*, IsEmpty};

pub trait MaterialVoxel {
    type Material: Eq;

    /// Used for comparing the materials of voxels. A single quad of the greedy mesh will contain
    /// faces of voxels that all share the same material.
    fn material(&self) -> Self::Material;
}

/// Contains the output from the `greedy_quads` algorithm. Can be reused to avoid re-allocations.
pub struct GreedyQuadsBuffer<M> {
    /// One group of quads per cube face.
    pub quad_groups: [QuadGroup<M>; 6],

    // A single array is used for the visited mask because it allows us to index by the same strides
    // as the voxels array. It also only requires a single allocation.
    visited: Array3<bool>,
}

impl<M> GreedyQuadsBuffer<M> {
    pub fn new(extent: Extent3i) -> Self {
        Self {
            quad_groups: QuadGroup::init_all_groups(),
            visited: Array3::fill(extent, false),
        }
    }

    pub fn reset(&mut self, extent: Extent3i) {
        for group in self.quad_groups.iter_mut() {
            group.quads.clear();
        }

        if extent.shape != self.visited.extent().shape {
            self.visited = Array3::fill(extent, false);
        }
        self.visited.set_minimum(extent.minimum);
    }

    /// Returns the total count of quads across all groups.
    pub fn num_quads(&self) -> usize {
        let mut sum = 0;
        for group in self.quad_groups.iter() {
            sum += group.quads.len();
        }

        sum
    }
}

/// Pads the given chunk extent with exactly the amount of space required for running the
/// `greedy_quads` algorithm.
pub fn padded_greedy_quads_chunk_extent(chunk_extent: &Extent3i) -> Extent3i {
    chunk_extent.padded(1)
}

/// The "Greedy Meshing" algorithm described by Mikola Lysenko in the
/// [0fps article](https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/).
///
/// All visible faces of voxels on the interior of `extent` will be part of some `Quad` returned via
/// the `output` buffer. A 3x3x3 kernel will be applied to each point on the interior, hence the
/// extra padding required on the boundary. `voxels` only needs to contain the set of points in
/// `extent`.
///
/// The quads can be post-processed into meshes as the user sees fit.
pub fn greedy_quads<V, T>(
    voxels: &V,
    extent: &Extent3i,
    output: &mut GreedyQuadsBuffer<T::Material>,
) where
    V: Array<[i32; 3]>
        + GetUncheckedRelease<Stride, T>
        + ForEach<[i32; 3], (Point3i, Stride), Data = T>,
    T: IsEmpty + MaterialVoxel,
{
    output.reset(*extent);
    let GreedyQuadsBuffer {
        visited,
        quad_groups,
    } = output;

    // Avoid accessing out of bounds with a 3x3x3 kernel.
    let Extent3i {
        shape: interior_shape,
        minimum: interior_min,
    } = extent.padded(-1);

    for group in quad_groups.iter_mut() {
        greedy_quads_for_group(voxels, interior_min, interior_shape, visited, group);
    }
}

fn greedy_quads_for_group<V, T>(
    voxels: &V,
    interior_min: Point3i,
    interior_shape: Point3i,
    visited: &mut Array3<bool>,
    quad_group: &mut QuadGroup<T::Material>,
) where
    V: Array<[i32; 3]>
        + GetUncheckedRelease<Stride, T>
        + ForEach<[i32; 3], (Point3i, Stride), Data = T>,
    T: IsEmpty + MaterialVoxel,
{
    visited.reset_values(false);

    let QuadGroup {
        quads,
        meta:
            QuadGroupMeta {
                n_sign,
                n,
                u,
                v,
                n_axis,
                u_axis,
                v_axis,
                ..
            },
    } = quad_group;

    let i_n = n_axis.index();
    let i_u = u_axis.index();
    let i_v = v_axis.index();

    let num_slices = interior_shape.at(i_n);
    let slice_shape = *n + *u * interior_shape.at(i_u) + *v * interior_shape.at(i_v);
    let mut slice_extent = Extent3i::from_min_and_shape(interior_min, slice_shape);

    let normal = *n * *n_sign;

    let visibility_offset = voxels.stride_from_local_point(&Local(normal));

    let u_stride = voxels.stride_from_local_point(&Local(*u));
    let v_stride = voxels.stride_from_local_point(&Local(*v));

    for _ in 0..num_slices {
        let slice_ub = slice_extent.least_upper_bound();
        let u_ub = slice_ub.at(i_u);
        let v_ub = slice_ub.at(i_v);

        voxels.for_each(&slice_extent, |(p, p_stride): (Point3i, Stride), voxel| {
            let quad_material = voxel.material();

            // These are the boundaries on quad width and height so it is contained in the slice.
            let mut max_width = u_ub - p.at(i_u);
            let max_height = v_ub - p.at(i_v);

            // Greedily search for the biggest visible quad that matches this material.
            //
            // Start by finding the widest quad in the U direction.
            let mut row_start_stride = p_stride;
            let quad_width = get_row_width(
                voxels,
                visited,
                &quad_material,
                visibility_offset,
                row_start_stride,
                u_stride,
                max_width,
            );

            if quad_width == 0 {
                // Not even the first face in the quad was visible.
                return;
            }

            // Now see how tall we can make the quad in the V direction without changing the width.
            max_width = max_width.min(quad_width);
            row_start_stride += v_stride;
            let mut quad_height = 1;
            while quad_height < max_height {
                let row_width = get_row_width(
                    voxels,
                    visited,
                    &quad_material,
                    visibility_offset,
                    row_start_stride,
                    u_stride,
                    max_width,
                );
                if row_width < quad_width {
                    break;
                }
                quad_height += 1;
                row_start_stride += v_stride;
            }

            quads.push((
                Quad {
                    minimum: p,
                    width: quad_width,
                    height: quad_height,
                },
                quad_material,
            ));

            // Mark the quad as visited.
            let quad_extent =
                Extent3i::from_min_and_shape(p, *n + *u * quad_width + *v * quad_height);
            visited.fill_extent(&quad_extent, true);
        });

        // Move to the next slice.
        slice_extent += *n;
    }
}

fn get_row_width<V, T>(
    voxels: &V,
    visited: &Array3<bool>,
    quad_material: &T::Material,
    visibility_offset: Stride,
    start_stride: Stride,
    delta_stride: Stride,
    max_width: i32,
) -> i32
where
    V: Array<[i32; 3]>
        + GetUncheckedRelease<Stride, T>
        + ForEach<[i32; 3], (Point3i, Stride), Data = T>,
    T: IsEmpty + MaterialVoxel,
{
    let mut quad_width = 0;
    let mut row_stride = start_stride;
    while quad_width < max_width {
        if visited.get_unchecked_release(row_stride) {
            // Already have a quad for this voxel face.
            break;
        }

        let voxel = voxels.get_unchecked_release(row_stride);
        if voxel.is_empty() || !voxel.material().eq(quad_material) {
            // Voxel needs to be non-empty and match the quad material.
            break;
        }

        let adjacent_voxel = voxels.get_unchecked_release(row_stride + visibility_offset);
        if !adjacent_voxel.is_empty() {
            // The adjacent voxel sharing this face must be empty for the face to be visible.
            break;
        }

        quad_width += 1;
        row_stride += delta_stride;
    }

    quad_width
}

#[derive(Clone, Copy)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    fn index(&self) -> usize {
        *self as usize
    }
}

/// A set of `Quad`s that share an orientation.
pub struct QuadGroup<M> {
    /// The quads themselves. We rely on the group's metadata to interpret them.
    pub quads: Vec<(Quad, M)>,
    pub meta: QuadGroupMeta,
}

impl<M> QuadGroup<M> {
    pub fn new(meta: QuadGroupMeta) -> Self {
        Self {
            quads: Vec::new(),
            meta,
        }
    }

    pub fn init_all_groups() -> [Self; 6] {
        // NOTE: As a heuristic, we intentionally avoid using Y as the U direction, because we
        // expect that X and Z will be longer for most voxel terrains.
        [
            //                                 +/- N        U        V
            Self::new(QuadGroupMeta::new(-1, Axis::X, Axis::Z, Axis::Y)),
            Self::new(QuadGroupMeta::new(-1, Axis::Y, Axis::X, Axis::Z)),
            Self::new(QuadGroupMeta::new(-1, Axis::Z, Axis::X, Axis::Y)),
            Self::new(QuadGroupMeta::new(1, Axis::X, Axis::Z, Axis::Y)),
            Self::new(QuadGroupMeta::new(1, Axis::Y, Axis::X, Axis::Z)),
            Self::new(QuadGroupMeta::new(1, Axis::Z, Axis::X, Axis::Y)),
        ]
    }
}

/// Metadata that's used to aid in the geometric calculations for one of the 6 possible cube faces.
pub struct QuadGroupMeta {
    // Determines whether we're looking at a positive or negative face of each cube.
    pub n_sign: i32,

    // Used for indexing.
    pub n_axis: Axis,
    pub u_axis: Axis,
    pub v_axis: Axis,

    // These vectors are always some permutation of +X, +Y, and +Z.
    pub n: Point3i,
    pub u: Point3i,
    pub v: Point3i,

    pub mesh_normal: Point3f,
}

impl QuadGroupMeta {
    pub fn new(n_sign: i32, n_axis: Axis, u_axis: Axis, v_axis: Axis) -> Self {
        let xyz = [PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])];

        let n = xyz[n_axis.index()];
        let mesh_normal: Point3f = (n * n_sign).into();

        Self {
            n_sign,

            n_axis,
            u_axis,
            v_axis,

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

    /// Returns the 6 vertex indices for the quad in order to make two triangles in a mesh.
    pub fn indices(&self, start: usize) -> [usize; 6] {
        match self.n_axis {
            Axis::X => quad_indices(start, self.n_sign > 0),
            Axis::Y => quad_indices(start, self.n_sign > 0),
            // Sign is intentionally flipped because sign of NUV permutation is different when using
            // our heuristic.
            Axis::Z => quad_indices(start, self.n_sign < 0),
        }
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
}

/// Returns the vertex indices for a single quad (two triangles). The triangles may have either
/// clockwise or counter-clockwise winding. `start` is the first index.
pub fn quad_indices(start: usize, clockwise: bool) -> [usize; 6] {
    if clockwise {
        [start, start + 2, start + 1, start + 1, start + 2, start + 3]
    } else {
        [start, start + 1, start + 2, start + 1, start + 3, start + 2]
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
    /// `QuadGroup::quad_corners`.
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
