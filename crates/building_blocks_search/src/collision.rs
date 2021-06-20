pub mod ball;
pub mod ray;

pub use ball::*;
pub use ray::*;

use building_blocks_core::prelude::*;

use nalgebra as na;

/// The result of a collision query against an `OctreeDbvt`.
#[derive(Clone, Debug)]
pub struct VoxelImpact<I> {
    /// The voxel point.
    pub point: Point3i,
    /// The impact type, which depends on the query.
    pub impact: I,
}

fn impact_with_leaf_octant(
    octant: &Octant,
    contact: &na::Point3<f32>,
    octant_normal: &na::Vector3<f32>,
) -> Point3i {
    if octant.edge_length() == 1 {
        octant.minimum()
    } else {
        // Octant is not a single voxel, so we need to calculate which voxel in the
        // octant was hit.
        //
        // Maybe converting the intersection coordinates to integers will not always
        // land in the correct voxel. It should help to nudge the point along the
        // intersection normal by some amount less than 1.0.
        const NUDGE_AMOUNT: f32 = 0.25;
        let nudged_p = contact - NUDGE_AMOUNT * octant_normal;

        Point3f::from(nudged_p).in_voxel()
    }
}

#[cfg(test)]
mod test_util {
    use crate::OctreeDbvt;

    use building_blocks_core::prelude::*;
    use building_blocks_storage::prelude::*;

    pub fn bvt_with_voxels_filled(fill_points: &[Point3i]) -> OctreeDbvt<i32> {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let mut voxels = Array3x1::fill(extent, Voxel(false));
        for &p in fill_points.iter() {
            *voxels.get_mut(p) = Voxel(true);
        }

        let octree = OctreeSet::from_array3(&voxels, *voxels.extent());
        let mut bvt = OctreeDbvt::default();
        let key = 0; // unimportant
        bvt.insert(key, octree);

        bvt
    }

    pub fn bvt_with_all_voxels_filled() -> OctreeDbvt<i32> {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let voxels = Array3x1::fill(extent, Voxel(true));

        let octree = OctreeSet::from_array3(&voxels, *voxels.extent());
        let mut bvt = OctreeDbvt::default();
        let key = 0; // unimportant
        bvt.insert(key, octree);

        bvt
    }

    #[derive(Clone)]
    pub struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }
}
