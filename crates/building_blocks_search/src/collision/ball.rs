use super::impact_with_leaf_octant;

use crate::{OctreeDbvt, OctreeDbvtVisitor, VoxelImpact};

use building_blocks_core::prelude::*;
use building_blocks_storage::prelude::VisitStatus;

use core::hash::Hash;
use nalgebra::{self as na, zero, Isometry3, Translation3, UnitQuaternion};
use ncollide3d::{
    bounding_volume::{BoundingVolume, HasBoundingVolume, AABB},
    query::{time_of_impact, DefaultTOIDispatcher, Ray, TOIStatus, TOI},
    shape::{Ball, Cuboid},
};

/// The impact of a ball with an `OctreeDbvt`.
pub type VoxelBallImpact = VoxelImpact<TOI<f32>>;

/// Casts a ball of `radius` along `ray` and returns the coordinates of the first voxel that intersects the ball. Voxels are
/// modeled as axis-aligned bounding boxes (AABBs).
///
/// `ray.dir` is the velocity vector of the ball, and any collisions that would occur after `max_toi` will not be considered.
///
/// `predicate` can be used to filter voxels by returning `false`.
pub fn cast_ball_at_voxels<K>(
    octree: &OctreeDbvt<K>,
    radius: f32,
    ray: Ray<f32>,
    max_toi: f32,
    predicate: impl Fn(Point3i) -> bool,
) -> Option<VoxelBallImpact>
where
    K: Eq + Hash,
{
    let mut visitor = VoxelSBallCast::new(radius, ray, max_toi, predicate);
    octree.visit(&mut visitor);

    visitor.earliest_impact
}

struct VoxelSBallCast<F> {
    earliest_impact: Option<VoxelImpact<TOI<f32>>>,
    ball: Ball<f32>,
    ball_start_isom: Isometry3<f32>,
    ray: Ray<f32>,
    max_toi: f32,
    ball_path_aabb: AABB<f32>,
    predicate: F,
}

impl<F> VoxelSBallCast<F> {
    fn new(radius: f32, ray: Ray<f32>, max_toi: f32, predicate: F) -> Self {
        let ball = Ball::new(radius);

        let start = ray.origin;
        let end = ray.point_at(max_toi);

        let ball_start_isom =
            Isometry3::from_parts(Translation3::from(start.coords), UnitQuaternion::identity());
        let ball_end_isom =
            Isometry3::from_parts(Translation3::from(end.coords), UnitQuaternion::identity());

        // Make an AABB that bounds the ball through its entire path.
        let ball_start_aabb: AABB<f32> = ball.bounding_volume(&ball_start_isom);
        let ball_end_aabb: AABB<f32> = ball.bounding_volume(&ball_end_isom);
        let ball_path_aabb = ball_start_aabb.merged(&ball_end_aabb);

        Self {
            earliest_impact: None,
            ball,
            ball_start_isom,
            ray,
            max_toi,
            ball_path_aabb,
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

impl<F> OctreeDbvtVisitor for VoxelSBallCast<F>
where
    F: Fn(Point3i) -> bool,
{
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_leaf: bool) -> VisitStatus {
        if !self.ball_path_aabb.intersects(aabb) {
            // The ball couldn't intersect any voxels in this AABB, because it doesn't even intersect the AABB that bounds the
            // ball's path.
            return VisitStatus::Stop;
        }

        if let Some(octant) = octant {
            // Cast a ball at this octant.
            let octant_extent = Extent3i::from(*octant);
            let voxel_velocity = na::Vector3::zeros();
            let target_distance = 0.0;
            if let Some(impact) = time_of_impact(
                &DefaultTOIDispatcher,
                &self.ball_start_isom,
                &self.ray.dir,
                &self.ball,
                &extent3i_cuboid_transform(&octant_extent),
                &voxel_velocity,
                &extent3i_cuboid(&octant_extent),
                self.max_toi,
                target_distance,
            )
            // Unsupported shape queries return Err
            .unwrap()
            {
                if impact.status != TOIStatus::Converged {
                    // Something bad happened with the TOI algorithm. Let's just keep going down this branch and hope it gets
                    // better. If we're at a leaf, we won't consider this a legitimate impact.
                    return VisitStatus::Continue;
                }

                if is_leaf && impact.toi < self.earliest_toi() {
                    // The contact point is the center of the ball plus the ball's "local witness."
                    let contact = self.ray.point_at(impact.toi) + impact.witness1.coords;

                    let point = impact_with_leaf_octant(octant, &contact, &impact.normal2);
                    if (self.predicate)(point) {
                        self.earliest_impact = Some(VoxelImpact { point, impact });
                    }
                }
            } else {
                // The ball won't intersect this octant.
                return VisitStatus::Stop;
            }
        }

        VisitStatus::Continue
    }
}

fn extent3i_cuboid(e: &Extent3i) -> Cuboid<f32> {
    Cuboid::new(half_extent(e.shape))
}

fn extent3i_cuboid_transform(e: &Extent3i) -> Isometry3<f32> {
    let min = na::Point3::from(Point3f::from(e.minimum));
    let center = min + half_extent(e.shape);

    Isometry3::new(center.coords, zero())
}

fn half_extent(shape: Point3i) -> na::Vector3<f32> {
    na::Vector3::from(Point3f::from(shape)) / 2.0
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

    use ncollide3d::na;

    #[test]
    fn ball_cast_hits_expected_voxel() {
        let bvt = bvt_with_voxels_filled(&[PointN([0, 0, 0]), PointN([0, 15, 0])]);

        // Cast ball at the corners.

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let radius = 0.5;

        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = cast_ball_at_voxels(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));

        let ray = Ray::new(start, na::Point3::new(0.0, 15.5, 0.0) - start);
        let result = cast_ball_at_voxels(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 15, 0]));

        // Cast into the middle where we shouldn't hit anything.

        let ray = Ray::new(start, na::Point3::new(0.0, 3.0, 0.0) - start);
        let result = cast_ball_at_voxels(&bvt, radius, ray, std::f32::INFINITY, |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn ball_cast_hits_expected_voxel_for_collapsed_leaf() {
        let bvt = bvt_with_all_voxels_filled();

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let radius = 0.5;
        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = cast_ball_at_voxels(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));
    }
}
