use super::PosNormMesh;

use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, ArrayExtentVisitor, GetUncheckedRelease};

pub trait Height {
    fn height(&self) -> f32;
}

impl Height for f32 {
    fn height(&self) -> f32 {
        *self
    }
}

/// Pads the given chunk extent with exactly the amount of space required for running the `triangulate_height_map` algorithm.
pub fn padded_height_map_chunk_extent(chunk_extent: &Extent2i) -> Extent2i {
    chunk_extent.padded(1).add_to_shape(PointN([1; 2]))
}

/// The output buffers used by `triangulate_height_map`. These buffers can be reused to avoid reallocating memory.
#[derive(Default)]
pub struct HeightMapMeshBuffer {
    /// The surface positions and normals. The normals are *not* normalized, since that is done most efficiently on the GPU.
    pub mesh: PosNormMesh,

    // Used to map back from voxel stride to vertex index.
    stride_to_index: Vec<u32>,
}

impl HeightMapMeshBuffer {
    /// Clears all of the buffers, but keeps the memory allocated for reuse.
    pub fn reset(&mut self, array_size: usize) {
        self.mesh.clear();

        // Just make sure this buffer is long enough, whether or not we've used it before.
        self.stride_to_index.resize(array_size, 0);
    }
}

/// Generates a mesh with a vertex at each point on the interior of `extent`. Surface normals are estimated using central
/// differencing, which requires each vertex to have a complete Von Neumann neighborhood. This means that points on the boundary
/// of `extent` are not eligible as mesh vertices, but they are still required.
///
/// This is illustrated in the ASCII art below, where "b" is a boundary point and "i" is an interior point. Line segments denote
/// the edges of the mesh.
///
/// ```text
/// b   b   b   b
///
/// b   i - i   b
///     | / |
/// b   i - i   b
///
/// b   b   b   b
/// ```
pub fn triangulate_height_map<A, H>(
    height_map: &A,
    extent: &Extent2i,
    output: &mut HeightMapMeshBuffer,
) where
    A: Array<[i32; 2]>
        + GetUncheckedRelease<Stride, H>
        + ForEach<[i32; 2], (Point2i, Stride), Data = H>,
    H: Height,
{
    output.reset(height_map.extent().num_points());

    // Avoid accessing out of bounds with a 3x3x3 kernel.
    let interior_extent = extent.padded(-1);

    let x_stride = height_map.stride_from_local_point(Local(PointN([1, 0])));
    let y_stride = height_map.stride_from_local_point(Local(PointN([0, 1])));

    height_map.for_each(
        &interior_extent,
        |(p, stride): (Point2i, Stride), height| {
            // Note: Although we use (x, y) for the coordinates of the height map, these should be considered (x, z) in world
            // coordinates, because +Y is the UP vector.
            let pz = p.y();
            let y = height.height();

            output.stride_to_index[stride.0] = output.mesh.positions.len() as u32;
            output.mesh.positions.push([p.x() as f32, y, pz as f32]);

            // Use central differencing to calculate the surface normal.
            //
            // From calculus, we know that gradients are always orthogonal to a level set. The surface approximated by the
            // height map h(x, z) happens to be the 0 level set of the function:
            //
            // f(x, y, z) = y - h(x, z)
            //
            // And the gradient is:
            //
            // grad f = [-dh/dx, 1, -dh/dz]
            let l_stride = stride - x_stride;
            let r_stride = stride + x_stride;
            let b_stride = stride - y_stride;
            let t_stride = stride + y_stride;
            let l_y = height_map.get_unchecked_release(l_stride).height();
            let r_y = height_map.get_unchecked_release(r_stride).height();
            let b_y = height_map.get_unchecked_release(b_stride).height();
            let t_y = height_map.get_unchecked_release(t_stride).height();
            let dy_dx = (r_y - l_y) / 2.0;
            let dy_dz = (t_y - b_y) / 2.0;
            // Not normalized, because that's done more efficiently on the GPU.
            output.mesh.normals.push([-dy_dx, 1.0, -dy_dz]);
        },
    );

    // Only add a quad when p is the bottom-left corner of a quad that fits in the interior.
    let quads_extent = interior_extent.add_to_shape(PointN([-1; 2]));

    let visitor = ArrayExtentVisitor::new_global(height_map.extent(), quads_extent);
    visitor.for_each_point_and_stride(|_p, bl_stride| {
        let br_stride = bl_stride + x_stride;
        let tl_stride = bl_stride + y_stride;
        let tr_stride = bl_stride + x_stride + y_stride;

        let bl_index = output.stride_to_index[bl_stride.0];
        let br_index = output.stride_to_index[br_stride.0];
        let tl_index = output.stride_to_index[tl_stride.0];
        let tr_index = output.stride_to_index[tr_stride.0];

        output
            .mesh
            .indices
            .extend_from_slice(&[bl_index, tl_index, tr_index, bl_index, tr_index, br_index]);
    });
}
