//! Acceleration structures and spatial queries on voxels.

pub mod octree;

pub use octree::{Octant, Octree, OctreeVisitor};

#[cfg(feature = "ncollide")]
pub mod collision;

#[cfg(feature = "ncollide")]
pub use collision::{voxel_ray_cast, voxel_sphere_cast};

#[cfg(feature = "ncollide")]
pub mod octree_dbvt;

#[cfg(feature = "ncollide")]
pub use octree_dbvt::{OctreeDBVT, OctreeDBVTVisitor};

#[cfg(feature = "ncollide")]
pub use ncollide3d;

#[cfg(feature = "ncollide")]
mod na_conversions {
    use building_blocks_core::{Point3f, Point3i};
    use nalgebra as na;

    pub fn na_point3f_from_point3i(p: Point3i) -> na::Point3<f32> {
        mint::Point3::from(Point3f::from(p)).into()
    }
}
