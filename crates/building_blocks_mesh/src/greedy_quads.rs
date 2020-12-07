use super::{
    quad::{OrientedCubeFace, UnorientedQuad},
    MaterialVoxel,
};

use building_blocks_core::{axis::Axis3Permutation, prelude::*};
use building_blocks_storage::{access::GetUncheckedRelease, prelude::*, IsEmpty};

/// Contains the output from the `greedy_quads` algorithm. Can be reused to avoid re-allocations.
pub struct GreedyQuadsBuffer<M> {
    /// One group of quads per cube face.
    pub quad_groups: [QuadGroup<M>; 6],

    // A single array is used for the visited mask because it allows us to index by the same strides
    // as the voxels array. It also only requires a single allocation.
    visited: Array3<bool>,
}

/// A set of `Quad`s that share an orientation. Each quad may specify a material of type `M`.
pub struct QuadGroup<M> {
    /// The quads themselves. We rely on the cube face metadata to interpret them.
    pub quads: Vec<(UnorientedQuad, M)>,
    /// One of 6 cube faces. All quads in this struct are comprised of only this face.
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

impl<M> GreedyQuadsBuffer<M> {
    pub fn new(extent: Extent3i) -> Self {
        let quad_groups = [
            QuadGroup::new(OrientedCubeFace::new(-1, Axis3Permutation::XZY)),
            QuadGroup::new(OrientedCubeFace::new(-1, Axis3Permutation::YXZ)),
            QuadGroup::new(OrientedCubeFace::new(-1, Axis3Permutation::ZXY)),
            QuadGroup::new(OrientedCubeFace::new(1, Axis3Permutation::XZY)),
            QuadGroup::new(OrientedCubeFace::new(1, Axis3Permutation::YXZ)),
            QuadGroup::new(OrientedCubeFace::new(1, Axis3Permutation::ZXY)),
        ];

        Self {
            quad_groups,
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

/// The "Greedy Meshing" algorithm described by Mikola Lysenko in the [0fps
/// article](https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/).
///
/// All visible faces of voxels on the interior of `extent` will be part of some `Quad` returned via the `output` buffer. A
/// 3x3x3 kernel will be applied to each point on the interior, hence the extra padding required on the boundary. `voxels` only
/// needs to contain the set of points in `extent`.
///
/// A single quad of the greedy mesh will contain faces of voxels that all share the same material. The quads can be
/// post-processed into meshes as the user sees fit.
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

    let Extent3i {
        shape: interior_shape,
        minimum: interior_min,
    } = extent.padded(-1); // Avoid accessing out of bounds with a 3x3x3 kernel.

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
        face:
            OrientedCubeFace {
                n_sign,
                permutation,
                n,
                u,
                v,
                ..
            },
    } = quad_group;

    let [n_axis, u_axis, v_axis] = permutation.axes();
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
                UnorientedQuad {
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
