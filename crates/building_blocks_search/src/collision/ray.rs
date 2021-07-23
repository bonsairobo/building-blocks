use super::impact_with_leaf_octant;

use crate::{OctreeDbvt, OctreeDbvtVisitor, VoxelImpact};

use building_blocks_core::prelude::*;
use building_blocks_storage::prelude::VisitStatus;

use core::hash::Hash;
use nalgebra::Isometry3;
use ncollide3d::{
    bounding_volume::AABB,
    query::{Ray, RayCast, RayIntersection},
};

/// The impact of a ray with an `OctreeDbvt`.
pub type VoxelRayImpact = VoxelImpact<RayIntersection<f32>>;

/// Casts a ray and returns the coordinates of the first voxel that intersects the ray. Voxels are modeled as axis-aligned
/// bounding boxes (AABBs).
///
/// `ray.dir` is the velocity vector of the ray, and any collisions that would occur after `max_toi` will not be considered.
///
/// `predicate` can be used to filter voxels by returning `false`.
pub fn cast_ray_at_voxels<K>(
    octree: &OctreeDbvt<K>,
    ray: Ray<f32>,
    max_toi: f32,
    predicate: impl Fn(Point3i) -> bool,
) -> Option<VoxelRayImpact>
where
    K: Eq + Hash,
{
    let mut visitor = VoxelRayCast::new(ray, max_toi, predicate);
    octree.visit(&mut visitor);

    visitor.earliest_impact
}

struct VoxelRayCast<F> {
    earliest_impact: Option<VoxelImpact<RayIntersection<f32>>>,
    ray: Ray<f32>,
    max_toi: f32,
    predicate: F,
}

impl<F> VoxelRayCast<F> {
    fn new(ray: Ray<f32>, max_toi: f32, predicate: F) -> Self {
        Self {
            earliest_impact: None,
            ray,
            max_toi,
            predicate,
        }
    }

    fn earliest_toi(&self) -> f32 {
        self.earliest_impact
            .as_ref()
            .map(|i| i.impact.toi)
            .unwrap_or(std::f32::INFINITY)
    }
}

impl<F> OctreeDbvtVisitor for VoxelRayCast<F>
where
    F: Fn(Point3i) -> bool,
{
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_leaf: bool) -> VisitStatus {
        let solid = true;
        if let Some(toi) = aabb.toi_with_ray(&Isometry3::identity(), &self.ray, self.max_toi, solid)
        {
            if toi < self.earliest_toi() {
                if is_leaf {
                    // This calculation is more expensive than just TOI, so we only do it for leaves.
                    let impact = aabb
                        .toi_and_normal_with_ray(
                            &Isometry3::identity(),
                            &self.ray,
                            self.max_toi,
                            true,
                        )
                        .unwrap();

                    let octant = octant.expect("All leaves are octants");
                    let point = impact_with_leaf_octant(
                        &octant,
                        &self.ray.point_at(impact.toi),
                        &impact.normal,
                    );

                    if (self.predicate)(point) {
                        self.earliest_impact = Some(VoxelImpact { point, impact });
                    }
                }

                VisitStatus::Continue
            } else {
                // The TOI with any voxels in this octant can't be earliest.
                VisitStatus::Stop
            }
        } else {
            // There's no impact with any voxels in this octant.
            VisitStatus::Stop
        }
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::test_util::*;

    use nalgebra as na;

    #[test]
    fn raycast_hits_expected_voxel() {
        let bvt = bvt_with_voxels_filled(&[PointN([0, 0, 0]), PointN([0, 15, 0])]);

        // Cast rays at the corners.

        let start = na::Point3::new(-1.0, -1.0, -1.0);

        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = cast_ray_at_voxels(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));

        let ray = Ray::new(start, na::Point3::new(0.0, 15.5, 0.0) - start);
        let result = cast_ray_at_voxels(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 15, 0]));

        // Cast into the middle where we shouldn't hit anything.

        let ray = Ray::new(start, na::Point3::new(0.0, 3.0, 0.0) - start);
        let result = cast_ray_at_voxels(&bvt, ray, std::f32::INFINITY, |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn raycast_hits_expected_voxel_for_collapsed_leaf() {
        let bvt = bvt_with_all_voxels_filled();

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = cast_ray_at_voxels(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));
    }
}
