use crate::{
    octree::{Octant, VisitStatus},
    octree_dbvt::{OctreeDBVT, OctreeDBVTVisitor},
};

use building_blocks_core::{prelude::*, voxel_containing_point3f};

use core::hash::Hash;
use ncollide3d::{
    bounding_volume::AABB,
    na::Isometry3,
    query::{Ray, RayCast},
};

/// Casts a ray and returns the coordinates of the first voxel that intersects the ray. Voxels are
/// modeled as axis-aligned bounding boxes (AABBs).
///
/// `predicate_fn` can be used to filter points by returning `false`.
pub fn nearest_voxel_ray_cast<K>(
    octree: &OctreeDBVT<K>,
    ray: &Ray<f32>,
    max_toi: f32,
    predicate_fn: impl Fn(Point3i) -> bool,
) -> Option<NearestVoxelRayCastResult>
where
    K: Eq + Hash,
{
    let mut visitor = NearestVoxelRayCast::new(*ray, max_toi, predicate_fn);
    octree.visit(&mut visitor);

    match (visitor.nearest_point, visitor.nearest_aabb) {
        (Some(point), Some(aabb)) => Some(NearestVoxelRayCastResult {
            point,
            aabb,
            toi: visitor.earliest_toi,
        }),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct NearestVoxelRayCastResult {
    /// The point of the first voxel to intersect the ray.
    pub point: Point3i,
    /// The axis-aligned bounding box of the voxel at `point`.
    pub aabb: AABB<f32>,
    /// The time of impact of the ray and voxel.
    pub toi: f32,
}

struct NearestVoxelRayCast<F> {
    pub earliest_toi: f32,
    pub nearest_aabb: Option<AABB<f32>>,
    pub nearest_point: Option<Point3i>,
    pub num_ray_casts: usize,
    pub ray: Ray<f32>,
    pub max_toi: f32,
    predicate_fn: F,
}

impl<F> NearestVoxelRayCast<F> {
    fn new(ray: Ray<f32>, max_toi: f32, predicate_fn: F) -> Self {
        Self {
            earliest_toi: std::f32::MAX,
            nearest_aabb: None,
            nearest_point: None,
            num_ray_casts: 0,
            ray,
            max_toi,
            predicate_fn,
        }
    }
}

impl<F> OctreeDBVTVisitor for NearestVoxelRayCast<F>
where
    F: Fn(Point3i) -> bool,
{
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_leaf: bool) -> VisitStatus {
        self.num_ray_casts += 1;
        if let Some(toi) = aabb.toi_with_ray(&Isometry3::identity(), &self.ray, self.max_toi, true)
        {
            if toi < self.earliest_toi {
                if is_leaf {
                    let octant = octant.unwrap();
                    let voxel_point = if octant.edge_length == 1 {
                        octant.minimum
                    } else {
                        // Octant is not a single voxel, so we need to calculate which voxel in the
                        // octant was hit.
                        //
                        // There is a concern that converting the intersection coordinates to
                        // integers will not always land in the correct voxel. It should help to
                        // nudge the point along the intersection normal by some epsilon.
                        let normal = aabb
                            .toi_and_normal_with_ray(
                                &Isometry3::identity(),
                                &self.ray,
                                self.max_toi,
                                true,
                            )
                            .unwrap()
                            .normal;
                        let intersection_p = self.ray.point_at(toi);
                        let nudge_p = intersection_p - std::f32::EPSILON * normal;

                        voxel_containing_point3f(&nudge_p.into())
                    };

                    if (self.predicate_fn)(voxel_point) {
                        self.earliest_toi = toi;
                        self.nearest_aabb = Some(aabb.clone());
                        self.nearest_point = Some(voxel_point);
                    }
                }

                return VisitStatus::Continue;
            }
        }

        VisitStatus::Stop
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

    use crate::octree::Octree;

    use building_blocks_storage::{prelude::*, IsEmpty};

    use ncollide3d::na;

    #[test]
    fn raycast_hits_expected_voxel() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let mut voxels = Array3::fill(extent, Voxel(false));

        // Set a couple of corners as targets.
        *voxels.get_mut(&PointN([0, 0, 0])) = Voxel(true);
        *voxels.get_mut(&PointN([0, 15, 0])) = Voxel(true);

        let power = 4;
        let octree = Octree::from_array3(power, &voxels);
        let mut bvt = OctreeDBVT::new();
        let key = 0; // unimportant
        bvt.insert(key, octree);

        // Cast rays at the corners.

        let start = na::Point3::new(-1.0, -1.0, -1.0);

        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = nearest_voxel_ray_cast(&bvt, &ray, std::f32::MAX, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));

        let ray = Ray::new(start, na::Point3::new(0.0, 15.5, 0.0) - start);
        let result = nearest_voxel_ray_cast(&bvt, &ray, std::f32::MAX, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 15, 0]));

        // Cast into the middle where we shouldn't hit anything.

        let ray = Ray::new(start, na::Point3::new(0.0, 3.0, 0.0) - start);
        let result = nearest_voxel_ray_cast(&bvt, &ray, std::f32::MAX, |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn raycast_hits_expected_voxel_for_collapsed_leaf() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let voxels = Array3::fill(extent, Voxel(true));

        let power = 4;
        let octree = Octree::from_array3(power, &voxels);
        let mut bvt = OctreeDBVT::new();
        let key = 0; // unimportant
        bvt.insert(key, octree);

        let start = na::Point3::new(-1.0, -1.0, -1.0);
        let ray = Ray::new(start, na::Point3::new(0.5, 0.5, 0.5) - start);
        let result = nearest_voxel_ray_cast(&bvt, &ray, std::f32::MAX, |_| true).unwrap();
        assert_eq!(result.point, PointN([0, 0, 0]));
    }

    #[derive(Clone)]
    struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }
}
