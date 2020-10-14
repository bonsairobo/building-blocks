//! Acceleration structures and spatial queries on voxels.

pub mod octree;

#[cfg(feature = "ncollide")]
pub mod collision;

#[cfg(feature = "ncollide")]
pub mod octree_dbvt;
