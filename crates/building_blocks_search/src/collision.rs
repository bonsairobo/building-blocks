use crate::octree_dbvt::{OctreeDbvt, OctreeDbvtVisitor};

use building_blocks_core::prelude::*;
use building_blocks_storage::{Octant, VisitStatus};

use core::hash::Hash;
use nalgebra::{self as na, zero, Isometry3, Translation3, UnitQuaternion};
use ncollide3d::{
    bounding_volume::{BoundingVolume, HasBoundingVolume, AABB},
    query::{time_of_impact, DefaultTOIDispatcher, Ray, RayCast, RayIntersection, TOIStatus, TOI},
    shape::{Ball, Cuboid},
};

/// The result of a collision query against an `OctreeDbvt`.
#[derive(Clone, Debug)]
pub struct VoxelImpact<I> {
    /// The voxel point.
    pub point: Point3i,
    /// The impact type, which depends on the query.
    pub impact: I,
}

/// The impact of a ray with an `OctreeDbvt`.
pub type VoxelRayImpact = VoxelImpact<RayIntersection<f32>>;
/// The impact of a sphere with an `OctreeDbvt`.
pub type VoxelSphereImpact = VoxelImpact<TOI<f32>>;

// ██████╗  █████╗ ██╗   ██╗
// ██╔══██╗██╔══██╗╚██╗ ██╔╝
// ██████╔╝███████║ ╚████╔╝
// ██╔══██╗██╔══██║  ╚██╔╝
// ██║  ██║██║  ██║   ██║
// ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝

/// Casts a ray and returns the coordinates of the first voxel that intersects the ray. Voxels are
/// modeled as axis-aligned bounding boxes (AABBs).
///
/// `ray.dir` is the velocity vector of the ray, and any collisions that would occur after `max_toi`
/// will not be considered.
///
/// `predicate` can be used to filter voxels by returning `false`.
pub fn voxel_ray_cast<K>(
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
                    // This calculation is more expensive than just TOI, so we only do it for
                    // leaves.
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
                        self.earliest_impact = Some(VoxelImpact { impact, point });
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

// ███████╗██████╗ ██╗  ██╗███████╗██████╗ ███████╗
// ██╔════╝██╔══██╗██║  ██║██╔════╝██╔══██╗██╔════╝
// ███████╗██████╔╝███████║█████╗  ██████╔╝█████╗
// ╚════██║██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗██╔══╝
// ███████║██║     ██║  ██║███████╗██║  ██║███████╗
// ╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝

/// Casts a sphere of `radius` along `ray` and returns the coordinates of the first voxel that
/// intersects the sphere. Voxels are modeled as axis-aligned bounding boxes (AABBs).
///
/// `ray.dir` is the velocity vector of the sphere, and any collisions that would occur after
/// `max_toi` will not be considered.
///
/// `predicate` can be used to filter voxels by returning `false`.
pub fn voxel_sphere_cast<K>(
    octree: &OctreeDbvt<K>,
    radius: f32,
    ray: Ray<f32>,
    max_toi: f32,
    predicate: impl Fn(Point3i) -> bool,
) -> Option<VoxelSphereImpact>
where
    K: Eq + Hash,
{
    let mut visitor = VoxelSphereCast::new(radius, ray, max_toi, predicate);
    octree.visit(&mut visitor);

    visitor.earliest_impact
}

struct VoxelSphereCast<F> {
    earliest_impact: Option<VoxelImpact<TOI<f32>>>,
    ball: Ball<f32>,
    ball_start_isom: Isometry3<f32>,
    ray: Ray<f32>,
    max_toi: f32,
    ball_path_aabb: AABB<f32>,
    predicate: F,
}

impl<F> VoxelSphereCast<F> {
    fn new(radius: f32, ray: Ray<f32>, max_toi: f32, predicate: F) -> Self {
        let ball = Ball::new(radius);

        let start = ray.origin;
        let end = ray.point_at(max_toi);

        let ball_start_isom =
            Isometry3::from_parts(Translation3::from(start.coords), UnitQuaternion::identity());
        let ball_end_isom =
            Isometry3::from_parts(Translation3::from(end.coords), UnitQuaternion::identity());

        // Make an AABB that bounds the sphere through its entire path.
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

impl<F> OctreeDbvtVisitor for VoxelSphereCast<F>
where
    F: Fn(Point3i) -> bool,
{
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_leaf: bool) -> VisitStatus {
        if !self.ball_path_aabb.intersects(aabb) {
            // The ball couldn't intersect any voxels in this AABB, because it doesn't even
            // intersect the AABB that bounds the ball's path.
            return VisitStatus::Stop;
        }

        if let Some(octant) = octant {
            // Cast a sphere at this octant.
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
                    // Something bad happened with the TOI algorithm. Let's just keep going down
                    // this branch and hope it gets better. If we're at a leaf, we won't consider
                    // this a legitimate impact.
                    return VisitStatus::Continue;
                }

                if is_leaf && impact.toi < self.earliest_toi() {
                    // The contact point is the center of the sphere plus the sphere's "local
                    // witness."
                    let contact = self.ray.point_at(impact.toi) + impact.witness1.coords;

                    let point = impact_with_leaf_octant(&octant, &contact, &impact.normal2);
                    if (self.predicate)(point) {
                        self.earliest_impact = Some(VoxelImpact { impact, point });
                    }
                }
            } else {
                // The sphere won't intersect this octant.
                return VisitStatus::Stop;
            }
        }

        VisitStatus::Continue
    }
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

    use building_blocks_storage::{prelude::*, IsEmpty, OctreeSet};

    use ncollide3d::na;

    #[test]
    fn raycast_hits_expected_voxel() {
        let bvt = bvt_with_voxels_filled(&[PointN([0, 0, 0]), PointN([0, 15, 0])]);

        // Cast rays at the corners.

        let start = na::Point3::new(-1.0, -1.0, -1.0);

        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = voxel_ray_cast(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));

        let ray = Ray::new(start, na::Point3::new(0.0, 15.5, 0.0) - start);
        let result = voxel_ray_cast(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 15, 0]));

        // Cast into the middle where we shouldn't hit anything.

        let ray = Ray::new(start, na::Point3::new(0.0, 3.0, 0.0) - start);
        let result = voxel_ray_cast(&bvt, ray, std::f32::INFINITY, |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn raycast_hits_expected_voxel_for_collapsed_leaf() {
        let bvt = bvt_with_all_voxels_filled();

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = voxel_ray_cast(&bvt, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));
    }

    #[test]
    fn sphere_cast_hits_expected_voxel() {
        let bvt = bvt_with_voxels_filled(&[PointN([0, 0, 0]), PointN([0, 15, 0])]);

        // Cast sphere at the corners.

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let radius = 0.5;

        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = voxel_sphere_cast(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));

        let ray = Ray::new(start, na::Point3::new(0.0, 15.5, 0.0) - start);
        let result = voxel_sphere_cast(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 15, 0]));

        // Cast into the middle where we shouldn't hit anything.

        let ray = Ray::new(start, na::Point3::new(0.0, 3.0, 0.0) - start);
        let result = voxel_sphere_cast(&bvt, radius, ray, std::f32::INFINITY, |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn sphere_cast_hits_expected_voxel_for_collapsed_leaf() {
        let bvt = bvt_with_all_voxels_filled();

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let radius = 0.5;
        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = voxel_sphere_cast(&bvt, radius, ray, std::f32::INFINITY, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));
    }

    fn bvt_with_voxels_filled(fill_points: &[Point3i]) -> OctreeDbvt<i32> {
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

    fn bvt_with_all_voxels_filled() -> OctreeDbvt<i32> {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let voxels = Array3x1::fill(extent, Voxel(true));

        let octree = OctreeSet::from_array3(&voxels, *voxels.extent());
        let mut bvt = OctreeDbvt::default();
        let key = 0; // unimportant
        bvt.insert(key, octree);

        bvt
    }

    #[derive(Clone)]
    struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }
}
