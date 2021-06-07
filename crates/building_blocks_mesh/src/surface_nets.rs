use super::PosNormMesh;

use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, ArrayForEach};

/// Pads the given chunk extent with exactly the amount of space required for running the `surface_nets` algorithm.
pub fn padded_surface_nets_chunk_extent(chunk_extent: &Extent3i) -> Extent3i {
    chunk_extent.padded(1)
}

/// The output buffers used by `surface_nets`. These buffers can be reused to avoid reallocating memory.
#[derive(Default)]
pub struct SurfaceNetsBuffer {
    /// The isosurface positions and normals. Parallel to `surface_points`. The normals are *not* normalized, since that is done
    /// most efficiently on the GPU.
    pub mesh: PosNormMesh,
    /// Global lattice coordinates of every voxel that intersects the isosurface.
    pub surface_points: Vec<Point3i>,
    /// Stride of every voxel that intersects the isosurface. Can be used for efficient post-processing.
    pub surface_strides: Vec<Stride>,

    // Used to map back from voxel stride to vertex index.
    stride_to_index: Vec<u32>,
}

impl SurfaceNetsBuffer {
    /// Clears all of the buffers, but keeps the memory allocated for reuse.
    pub fn reset(&mut self, array_size: usize) {
        self.mesh.clear();
        self.surface_points.clear();
        self.surface_strides.clear();

        // Just make sure this buffer is big enough, whether or not we've used it before.
        self.stride_to_index.resize(array_size, 0);
    }
}

/// The Naive Surface Nets smooth voxel meshing algorithm.
///
/// This is basically just dual contouring a uniform grid with:
///   - positions estimated as the centroid of cube edge crossings
///   - surface normals estimated with central differencing
///
/// Extracts an isosurface mesh from the [signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function) `sdf`.
/// Each value in the field determines how close that point is to the isosurface. Negative values are considered "interior" of
/// the surface volume, and positive values are considered "exterior." These lattice points will be considered corners of unit
/// cubes. For each unit cube, at most one isosurface vertex will be estimated, as below, where `p` is a positive corner value,
/// `n` is a negative corner value, `s` is an isosurface vertex, and `|` or `-` are mesh polygons connecting the vertices.
///
/// ```text
/// p   p   p   p
///   s---s
/// p | n | p   p
///   s   s---s
/// p | n   n | p
///   s---s---s
/// p   p   p   p
/// ```
///
/// The set of corners sampled is exactly the set of points in `extent`. `sdf` must contain all of those points.
pub fn surface_nets<A, T>(
    sdf: &A,
    extent: &Extent3i,
    voxel_size: f32,
    output: &mut SurfaceNetsBuffer,
) where
    A: IndexedArray<[i32; 3]> + Get<Stride, Item = T>,
    T: SignedDistance,
{
    output.reset(sdf.extent().num_points());

    estimate_surface(sdf, extent, voxel_size, output);
    make_all_quads(sdf, extent, output);
}

// Find all vertex positions and normals. Also generate a map from grid position to vertex index to be used to look up vertices
// when generating quads.
fn estimate_surface<A, T>(
    sdf: &A,
    extent: &Extent3i,
    voxel_size: f32,
    output: &mut SurfaceNetsBuffer,
) where
    A: IndexedArray<[i32; 3]> + Get<Stride, Item = T>,
    T: SignedDistance,
{
    // Precalculate these offsets to do faster linear indexing.
    let mut corner_offset_strides = [Stride(0); 8];
    let corner_offsets = Local::localize_points_array(&Point3i::CUBE_CORNER_OFFSETS);
    sdf.strides_from_local_points(&corner_offsets, &mut corner_offset_strides);

    // Avoid accessing out of bounds with a 2x2x2 kernel.
    let iter_extent = extent.add_to_shape(Point3i::fill(-1));

    let visitor = ArrayForEach::new_global(*sdf.extent(), iter_extent);
    visitor.for_each(|p, p_stride| {
        // Get the corners of the cube with minimal corner p.
        let mut corner_strides = [Stride(0); 8];
        for i in 0..8 {
            corner_strides[i] = p_stride + corner_offset_strides[i];
        }

        if let Some((position, normal)) =
            estimate_surface_in_cube(sdf, voxel_size, &p, &corner_strides)
        {
            output.stride_to_index[p_stride.0] = output.mesh.positions.len() as u32;
            output.surface_points.push(p);
            output.surface_strides.push(p_stride);
            output.mesh.positions.push(position);
            output.mesh.normals.push(normal);
        }
    });
}

const CUBE_EDGES: [[usize; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b011, 0b111],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b101, 0b111],
    [0b110, 0b111],
];

// Consider the grid-aligned cube where `point` is the minimal corner. Find a point inside this cube that is approximately on
// the isosurface.
//
// This is done by estimating, for each cube edge, where the isosurface crosses the edge (if it does at all). Then the estimated
// surface point is the average of these edge crossings.
fn estimate_surface_in_cube<A, T>(
    sdf: &A,
    voxel_size: f32,
    cube_min_corner: &Point3i,
    corner_strides: &[Stride],
) -> Option<([f32; 3], [f32; 3])>
where
    A: Get<Stride, Item = T>,
    T: SignedDistance,
{
    // Get the signed distance values at each corner of this cube.
    let mut corner_dists = [0.0; 8];
    let mut num_negative = 0;
    for (i, dist) in corner_dists.iter_mut().enumerate() {
        let d = sdf.get(corner_strides[i]).into();
        *dist = d;
        if d < 0.0 {
            num_negative += 1;
        }
    }

    if num_negative == 0 || num_negative == 8 {
        // No crossings.
        return None;
    }

    let centroid = centroid_of_edge_intersections(&corner_dists);
    let position = voxel_size * (Point3f::from(*cube_min_corner) + centroid + Point3f::fill(0.5));
    let normal = sdf_gradient(&corner_dists, &centroid);

    Some((position.0, normal))
}

fn centroid_of_edge_intersections(dists: &[f32; 8]) -> Point3f {
    let mut count = 0;
    let mut sum = Point3f::ZERO;
    for [corner1, corner2] in CUBE_EDGES.iter() {
        let d1 = dists[*corner1];
        let d2 = dists[*corner2];
        if (d1 < 0.0) != (d2 < 0.0) {
            count += 1;
            sum += estimate_surface_edge_intersection(*corner1, *corner2, d1, d2);
        }
    }

    sum / count as f32
}

// Given two cube corners, find the point between them where the SDF is zero. (This might not exist).
fn estimate_surface_edge_intersection(
    corner1: usize,
    corner2: usize,
    value1: f32,
    value2: f32,
) -> Point3f {
    let interp1 = value1 / (value1 - value2);
    let interp2 = 1.0 - interp1;

    PointN([
        (corner1 & 1) as f32 * interp2 + (corner2 & 1) as f32 * interp1,
        ((corner1 >> 1) & 1) as f32 * interp2 + ((corner2 >> 1) & 1) as f32 * interp1,
        ((corner1 >> 2) & 1) as f32 * interp2 + ((corner2 >> 2) & 1) as f32 * interp1,
    ])
}

/// Calculate the normal as the gradient of the distance field. Don't bother making it a unit vector, since we'll do that on the
/// GPU.
///
/// For each dimension, there are 4 cube edges along that axis. This will do bilinear interpolation between the differences
/// along those edges based on the position of the surface (s).
fn sdf_gradient(dists: &[f32; 8], s: &Point3f) -> [f32; 3] {
    let nx = 1.0 - s.x();
    let ny = 1.0 - s.y();
    let nz = 1.0 - s.z();

    let dx_z0 = ny * (dists[0b001] - dists[0b000]) + s.y() * (dists[0b011] - dists[0b010]);
    let dx_z1 = ny * (dists[0b101] - dists[0b100]) + s.y() * (dists[0b111] - dists[0b110]);
    let dx = nz * dx_z0 + s.z() * dx_z1;

    let dy_x0 = nz * (dists[0b010] - dists[0b000]) + s.z() * (dists[0b110] - dists[0b100]);
    let dy_x1 = nz * (dists[0b011] - dists[0b001]) + s.z() * (dists[0b111] - dists[0b101]);
    let dy = nx * dy_x0 + s.x() * dy_x1;

    let dz_y0 = nx * (dists[0b100] - dists[0b000]) + s.x() * (dists[0b101] - dists[0b001]);
    let dz_y1 = nx * (dists[0b110] - dists[0b010]) + s.x() * (dists[0b111] - dists[0b011]);
    let dz = ny * dz_y0 + s.y() * dz_y1;

    [dx, dy, dz]
}

// For every edge that crosses the isosurface, make a quad between the "centers" of the four cubes touching that surface. The
// "centers" are actually the vertex positions found earlier. Also, make sure the triangles are facing the right way. See the
// comments on `maybe_make_quad` to help with understanding the indexing.
fn make_all_quads<A, T>(sdf: &A, extent: &Extent3i, output: &mut SurfaceNetsBuffer)
where
    A: IndexedArray<[i32; 3]> + Get<Stride, Item = T>,
    T: SignedDistance,
{
    let mut xyz_strides = [Stride(0); 3];
    let xyz = [
        Local(PointN([1, 0, 0])),
        Local(PointN([0, 1, 0])),
        Local(PointN([0, 0, 1])),
    ];
    sdf.strides_from_local_points(&xyz, &mut xyz_strides);

    // NOTE: The checks against max prevent us from making quads on the 3 maximal planes of the grid. This is necessary to avoid
    // redundant quads when meshing adjacent chunks (assuming this will be used on a chunked voxel grid).
    let min = extent.minimum;
    let max = extent.max();

    for (p, p_stride) in output
        .surface_points
        .iter()
        .zip(output.surface_strides.iter())
    {
        // Do edges parallel with the X axis
        if p.y() != min.y() && p.z() != min.z() && p.x() != max.x() {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.mesh.positions,
                *p_stride,
                *p_stride + xyz_strides[0],
                xyz_strides[1],
                xyz_strides[2],
                &mut output.mesh.indices,
            );
        }
        // Do edges parallel with the Y axis
        if p.x() != min.x() && p.z() != min.z() && p.y() != max.y() {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.mesh.positions,
                *p_stride,
                *p_stride + xyz_strides[1],
                xyz_strides[2],
                xyz_strides[0],
                &mut output.mesh.indices,
            );
        }
        // Do edges parallel with the Z axis
        if p.x() != min.x() && p.y() != min.y() && p.z() != max.z() {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.mesh.positions,
                *p_stride,
                *p_stride + xyz_strides[2],
                xyz_strides[0],
                xyz_strides[1],
                &mut output.mesh.indices,
            );
        }
    }
}

// Construct a quad in the dual graph of the SDF lattice.
//
// The surface point s was found somewhere inside of the cube with minimal corner p1.
//
//       x ---- x
//      /      /|
//     x ---- x |
//     |   s  | x
//     |      |/
//    p1 --- p2
//
// And now we want to find the quad between p1 and p2 where s is a corner of the quad.
//
//          s
//         /|
//        / |
//       |  |
//   p1  |  |  p2
//       | /
//       |/
//
// If A is (of the three grid axes) the axis between p1 and p2,
//
//       A
//   p1 ---> p2
//
// then we must find the other 3 quad corners by moving along the other two axes (those orthogonal to A) in the negative
// directions; these are axis B and axis C.
fn maybe_make_quad<A, T>(
    sdf: &A,
    stride_to_index: &[u32],
    positions: &[[f32; 3]],
    p1: Stride,
    p2: Stride,
    axis_b_stride: Stride,
    axis_c_stride: Stride,
    indices: &mut Vec<u32>,
) where
    A: Get<Stride, Item = T>,
    T: SignedDistance,
{
    let d1 = sdf.get(p1);
    let d2 = sdf.get(p2);
    let negative_face = match (d1.is_negative(), d2.is_negative()) {
        (true, false) => false,
        (false, true) => true,
        _ => return, // No face.
    };

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = stride_to_index[p1.0];
    let v2 = stride_to_index[(p1 - axis_b_stride).0];
    let v3 = stride_to_index[(p1 - axis_c_stride).0];
    let v4 = stride_to_index[(p1 - axis_b_stride - axis_c_stride).0];
    let (pos1, pos2, pos3, pos4) = (
        positions[v1 as usize],
        positions[v2 as usize],
        positions[v3 as usize],
        positions[v4 as usize],
    );
    // Split the quad along the shorter axis, rather than the longer one.
    let quad = if sq_dist(pos1, pos4) < sq_dist(pos2, pos3) {
        if negative_face {
            [v1, v4, v2, v1, v3, v4]
        } else {
            [v1, v2, v4, v1, v4, v3]
        }
    } else if negative_face {
        [v2, v3, v4, v2, v1, v3]
    } else {
        [v2, v4, v3, v2, v3, v1]
    };
    indices.extend_from_slice(&quad);
}

fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}
