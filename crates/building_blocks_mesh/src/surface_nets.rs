//! The Naive Surface Nets smooth voxel meshing algorithm.
//!
//! For an in-depth explanation of the algorithm, read [here](https://medium.com/@bonsairobo/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14).
//!
//! The `surface_nets` function is designed to be used with a `ChunkMap3`, such that each chunk will
//! have its own mesh. In order to update the mesh for a chunk, you must copy not only the chunk,
//! but also the adjacent points, into an array and pass it to `surface_nets`.
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//! use building_blocks_mesh::surface_nets::*;
//!
//! use std::collections::HashSet;
//!
//! let chunk_shape = PointN([16; 3]);
//! let mut map = ChunkMap3::new(chunk_shape, 100.0, (), FastLz4 { level: 10 });
//!
//! // Mutate one or more of the chunks...
//! let mutated_chunk_keys = [PointN([0; 3]), PointN([16; 3])];
//!
//! // For each mutated chunk, and any adjacent chunk, the mesh will need to be updated. (This is
//! // not 100% true, but it's a conservative assumption to make. In reality, if a given voxel is
//! // mutated and it's L1 distance from an adjacent chunk is less than or equal to 2, that adjacent
//! // chunk's mesh is dependent on that voxel, so it must be re-meshed).
//! let mut chunk_keys_to_update: HashSet<Point3i> = HashSet::new();
//! let offsets = Point3i::moore_offsets();
//! for chunk_key in mutated_chunk_keys.into_iter() {
//!     chunk_keys_to_update.insert(*chunk_key);
//!     for offset in offsets.iter() {
//!         chunk_keys_to_update.insert(*chunk_key + *offset * chunk_shape);
//!     }
//! }
//!
//! // Now we generate mesh vertices for each chunk.
//! let local_cache = LocalChunkCache::new();
//! let reader = ChunkMapReader3::new(&map, &local_cache);
//! for chunk_key in chunk_keys_to_update.into_iter() {
//!     // It's crucial that we pad the chunk so we have access to adjacent points during meshing.
//!     let padded_chunk_extent = padded_surface_nets_chunk_extent(
//!         &map.extent_for_chunk_at_key(&chunk_key)
//!     );
//!     let mut padded_chunk = Array3::fill(padded_chunk_extent, 0.0);
//!     copy_extent(&padded_chunk_extent, &reader, &mut padded_chunk);
//!
//!     let mut sn_buffer = SurfaceNetsBuffer::default();
//!     surface_nets(&padded_chunk, &padded_chunk_extent, &mut sn_buffer);
//!     // Do something with the mesh output...
//! }
//! ```

use super::PosNormMesh;

use building_blocks_core::prelude::*;
use building_blocks_storage::{access::GetUncheckedRefRelease, prelude::*};

// ███████╗██╗   ██╗██████╗ ███████╗ █████╗  ██████╗███████╗
// ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝
// ███████╗██║   ██║██████╔╝█████╗  ███████║██║     █████╗
// ╚════██║██║   ██║██╔══██╗██╔══╝  ██╔══██║██║     ██╔══╝
// ███████║╚██████╔╝██║  ██║██║     ██║  ██║╚██████╗███████╗
// ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝

/// Pads the given chunk extent with exactly the amount of space required for running the
/// `surface_nets` algorithm.
pub fn padded_surface_nets_chunk_extent(chunk_extent: &Extent3i) -> Extent3i {
    chunk_extent.padded(1)
}

pub trait SignedDistance {
    fn distance(&self) -> f32;
}

impl SignedDistance for f32 {
    fn distance(&self) -> f32 {
        *self
    }
}

/// The output buffers used by `surface_nets`. These buffers can be reused to avoid reallocating
/// memory.
#[derive(Default)]
pub struct SurfaceNetsBuffer {
    /// The isosurface positions and normals. Parallel to `surface_points`. The normals are *not*
    /// normalized, since that is done most efficiently on the GPU.
    pub mesh: PosNormMesh,
    /// Global lattice coordinates of every voxel that intersects the isosurface.
    pub surface_points: Vec<Point3i>,
    /// Stride of every voxel that intersects the isosurface. Can be used with the
    /// `material_weights` function.
    pub surface_strides: Vec<Stride>,

    // Used to map back from voxel stride to vertex index.
    stride_to_index: Vec<usize>,
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

/// Extracts an isosurface mesh from the signed distance field `sdf`. The `sdf` describes a 3D
/// lattice of values. These lattice points will be considered corners of unit cubes. For each unit
/// cube, a single isosurface vertex will be estimated, as below, where "c" is a cube corner and "s"
/// is an isosurface vertex.
///
/// ```text
/// c - c - c - c
/// | s | s | s |
/// c - c - c - c
/// | s | s | s |
/// c - c - c - c
/// | s | s | s |
/// c - c - c - c
/// ```
///
/// The set of corners sampled is exactly the set of points in `extent`. `sdf` must contain all of
/// those points.
pub fn surface_nets<V, T>(sdf: &V, extent: &Extent3i, output: &mut SurfaceNetsBuffer)
where
    V: Array<[i32; 3]> + GetUncheckedRefRelease<Stride, T>,
    T: SignedDistance,
{
    output.reset(sdf.extent().num_points());

    estimate_surface(sdf, extent, output);
    make_all_quads(sdf, &extent, output);
}

// Find all vertex positions and normals. Also generate a map from grid position to vertex index
// to be used to look up vertices when generating quads.
fn estimate_surface<V, T>(sdf: &V, extent: &Extent3i, output: &mut SurfaceNetsBuffer)
where
    V: Array<[i32; 3]> + GetUncheckedRefRelease<Stride, T>,
    T: SignedDistance,
{
    // Precalculate these offsets to do faster linear indexing.
    let mut corner_offset_strides = [Stride(0); 8];
    let corner_offsets = Point3i::corner_offsets();
    sdf.strides_from_points(&corner_offsets, &mut corner_offset_strides);

    // Avoid accessing out of bounds with a 2x2x2 kernel.
    let iter_extent = extent.add_to_shape(PointN([-1; 3]));

    V::for_each_point_and_stride(sdf.extent(), &iter_extent, |p, p_stride| {
        // Get the corners of the cube with minimal corner p.
        let mut corner_strides = [Stride(0); 8];
        for i in 0..8 {
            corner_strides[i] = p_stride + corner_offset_strides[i];
        }

        if let Some((position, normal)) = estimate_surface_in_voxel(sdf, &p, &corner_strides) {
            output.stride_to_index[p_stride.0] = output.mesh.positions.len();
            output.surface_points.push(p);
            output.surface_strides.push(p_stride);
            output.mesh.positions.push(position);
            output.mesh.normals.push(normal);
        }
    });
}

const CUBE_EDGES: [(usize, usize); 12] = [
    (0b000, 0b001),
    (0b000, 0b010),
    (0b000, 0b100),
    (0b001, 0b011),
    (0b001, 0b101),
    (0b010, 0b011),
    (0b010, 0b110),
    (0b011, 0b111),
    (0b100, 0b101),
    (0b100, 0b110),
    (0b101, 0b111),
    (0b110, 0b111),
];

// Consider the grid-aligned cube where `point` is the minimal corner. Find a point inside this cube
// that is approximately on the isosurface.
//
// This is done by estimating, for each cube edge, where the isosurface crosses the edge (if it
// does at all). Then the estimated surface point is the average of these edge crossings.
fn estimate_surface_in_voxel<V, T>(
    sdf: &V,
    point: &Point3i,
    corner_strides: &[Stride],
) -> Option<([f32; 3], [f32; 3])>
where
    V: GetUncheckedRefRelease<Stride, T>,
    T: SignedDistance,
{
    // Get the signed distance values at each corner of this cube.
    let mut dists = [0.0; 8];
    let mut num_negative = 0;
    for (i, dist) in dists.iter_mut().enumerate() {
        let d = sdf.get_unchecked_ref_release(corner_strides[i]).distance();
        *dist = d;
        if d < 0.0 {
            num_negative += 1;
        }
    }

    if num_negative == 0 || num_negative == 8 {
        // No crossings.
        return None;
    }

    let mut count = 0;
    let mut sum = [0.0, 0.0, 0.0];
    for (offset1, offset2) in CUBE_EDGES.iter() {
        if let Some(intersection) =
            estimate_surface_edge_intersection(*offset1, *offset2, dists[*offset1], dists[*offset2])
        {
            count += 1;
            sum[0] += intersection[0];
            sum[1] += intersection[1];
            sum[2] += intersection[2];
        }
    }

    // Calculate the normal as the gradient of the distance field. Use central differencing. Don't
    // bother making it a unit vector, since we'll do that on the GPU.
    let normal_x = (dists[0b001] + dists[0b011] + dists[0b101] + dists[0b111])
        - (dists[0b000] + dists[0b010] + dists[0b100] + dists[0b110]);
    let normal_y = (dists[0b010] + dists[0b011] + dists[0b110] + dists[0b111])
        - (dists[0b000] + dists[0b001] + dists[0b100] + dists[0b101]);
    let normal_z = (dists[0b100] + dists[0b101] + dists[0b110] + dists[0b111])
        - (dists[0b000] + dists[0b001] + dists[0b010] + dists[0b011]);

    Some((
        [
            sum[0] / count as f32 + point.x() as f32 + 0.5,
            sum[1] / count as f32 + point.y() as f32 + 0.5,
            sum[2] / count as f32 + point.z() as f32 + 0.5,
        ],
        [normal_x, normal_y, normal_z],
    ))
}

// Given two cube corners, find the point between them where the SDF is zero.
// (This might not exist).
fn estimate_surface_edge_intersection(
    offset1: usize,
    offset2: usize,
    value1: f32,
    value2: f32,
) -> Option<[f32; 3]> {
    if (value1 < 0.0) == (value2 < 0.0) {
        return None;
    }

    let interp1 = value1 / (value1 - value2);
    let interp2 = 1.0 - interp1;
    let position = [
        (offset1 & 1) as f32 * interp2 + (offset2 & 1) as f32 * interp1,
        ((offset1 >> 1) & 1) as f32 * interp2 + ((offset2 >> 1) & 1) as f32 * interp1,
        ((offset1 >> 2) & 1) as f32 * interp2 + ((offset2 >> 2) & 1) as f32 * interp1,
    ];

    Some(position)
}

// For every edge that crosses the isosurface, make a quad between the "centers" of the four cubes
// touching that surface. The "centers" are actually the vertex positions found earlier. Also,
// make sure the triangles are facing the right way. See the comments on `maybe_make_quad` to help
// with understanding the indexing.
fn make_all_quads<V, T>(sdf: &V, extent: &Extent3i, output: &mut SurfaceNetsBuffer)
where
    V: Array<[i32; 3]> + GetUncheckedRefRelease<Stride, T>,
    T: SignedDistance,
{
    let mut xyz_strides = [Stride(0); 3];
    let xyz = [PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])];
    sdf.strides_from_points(&xyz, &mut xyz_strides);

    // NOTE: The checks against max prevent us from making quads on the 3 maximal planes of the
    // grid. This is necessary to avoid redundant quads when meshing adjacent chunks (assuming this
    // will be used on a chunked voxel grid).
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

// This is where the "dual" nature of surface nets comes into play.
//
// The surface point s was found somewhere inside of the cube "at" point p1.
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
// then we must find the other 3 quad corners by moving along the other two axes (those orthogonal
// to A) in the negative directions; these are axis B and axis C.
fn maybe_make_quad<V, T>(
    sdf: &V,
    stride_to_index: &[usize],
    positions: &[[f32; 3]],
    p1: Stride,
    p2: Stride,
    axis_b_stride: Stride,
    axis_c_stride: Stride,
    indices: &mut Vec<usize>,
) where
    V: GetUncheckedRefRelease<Stride, T>,
    T: SignedDistance,
{
    let voxel1 = sdf.get_unchecked_ref_release(p1);
    let voxel2 = sdf.get_unchecked_ref_release(p2);

    let face_result = is_face(voxel1.distance(), voxel2.distance());

    if let FaceResult::NoFace = face_result {
        return;
    }

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = stride_to_index[p1.0];
    let v2 = stride_to_index[(p1 - axis_b_stride).0];
    let v3 = stride_to_index[(p1 - axis_c_stride).0];
    let v4 = stride_to_index[(p1 - axis_b_stride - axis_c_stride).0];
    let (pos1, pos2, pos3, pos4) = (positions[v1], positions[v2], positions[v3], positions[v4]);
    // Split the quad along the shorter axis, rather than the longer one.
    let quad = if sq_dist(pos1, pos4) < sq_dist(pos2, pos3) {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => [v1, v2, v4, v1, v4, v3],
            FaceResult::FaceNegative => [v1, v4, v2, v1, v3, v4],
        }
    } else {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => [v2, v4, v3, v2, v3, v1],
            FaceResult::FaceNegative => [v2, v3, v4, v2, v1, v3],
        }
    };
    indices.extend_from_slice(&quad);
}

fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

enum FaceResult {
    NoFace,
    FacePositive,
    FaceNegative,
}

// Determine if the sign of the SDF flips between p1 and p2
fn is_face(d1: f32, d2: f32) -> FaceResult {
    match (d1 < 0.0, d2 < 0.0) {
        (true, false) => FaceResult::FacePositive,
        (false, true) => FaceResult::FaceNegative,
        _ => FaceResult::NoFace,
    }
}

// ███╗   ███╗ █████╗ ████████╗███████╗██████╗ ██╗ █████╗ ██╗     ███████╗
// ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗██║██╔══██╗██║     ██╔════╝
// ██╔████╔██║███████║   ██║   █████╗  ██████╔╝██║███████║██║     ███████╗
// ██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  ██╔══██╗██║██╔══██║██║     ╚════██║
// ██║ ╚═╝ ██║██║  ██║   ██║   ███████╗██║  ██║██║██║  ██║███████╗███████║
// ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚══════╝

pub trait MaterialVoxel {
    fn material_index(&self) -> usize;
}

/// Returns the material weights for each of the points in `surface_strides`.
///
/// Uses a 2x2x2 kernel (the same shape as the Surface Nets kernel) to average the adjacent
/// materials for each surface point. `voxels` should at least contain the extent that was used with
/// `surface_nets` in order to generate `surface_strides`. Currently limited to 4 materials per
/// surface chunk.
pub fn material_weights<V, T>(voxels: &V, surface_strides: &[Stride]) -> Vec<[f32; 4]>
where
    V: Array<[i32; 3]> + GetUncheckedRefRelease<Stride, T>,
    T: MaterialVoxel + SignedDistance,
{
    // Precompute the offsets for cube corners.
    let mut corner_offset_strides = [Stride(0); 8];
    let corner_offsets = Point3i::corner_offsets();
    voxels.strides_from_points(&corner_offsets, &mut corner_offset_strides);

    let mut material_weights = vec![[0.0; 4]; surface_strides.len()];

    for (i, p_stride) in surface_strides.iter().enumerate() {
        let w = &mut material_weights[i];
        for offset_stride in corner_offset_strides.iter() {
            let q_stride = *p_stride + *offset_stride;
            let voxel = voxels.get_unchecked_ref_release(q_stride);
            if voxel.distance() < 0.0 {
                let material_w = MATERIAL_WEIGHT_TABLE[voxel.material_index()];
                w[0] += material_w[0];
                w[1] += material_w[1];
                w[2] += material_w[2];
                w[3] += material_w[3];
            }
        }
    }

    material_weights
}

// Currently limited to 4 numbers for material weights.
const MATERIAL_WEIGHT_TABLE: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
