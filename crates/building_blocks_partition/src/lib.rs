//! Acceleration structures and spatial queries on voxels.

pub mod octree;

pub use octree::{FuncVisitor, Octant, Octree, OctreeVisitor};

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
