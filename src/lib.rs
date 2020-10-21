//! Data types, collections, and algorithms for working with maps on 2D and 3D integer lattices.
//! Commonly known as pixel or voxel data.
//!
//! This library is organized into several crates:
//! - **core**: lattice point and extent data types
//! - **image**: conversion of 2D lattice maps to/from images
//! - **mesh**: 3D isosurface generation algorithms, smooth and cubic
//! - **partition**: spatial queries on voxels, e.g. raycasting
//! - **procgen**: procedural generation of lattice maps, including sampled SDFs and height maps
//! - **search**: search algorithms on lattice maps
//! - **storage**: compressed storage for lattice maps, i.e. functions defined on `Z^2` and `Z^3`
//! - **vox**: conversion of 3D lattice maps to/from VOX data format

pub use building_blocks_core as core;
pub use building_blocks_storage as storage;

pub mod prelude {
    pub use super::core::prelude::*;
    pub use super::storage::prelude::*;
}

#[cfg(feature = "image")]
pub use building_blocks_image as image;

#[cfg(feature = "mesh")]
pub use building_blocks_mesh as mesh;

#[cfg(feature = "partition")]
pub use building_blocks_partition as partition;

#[cfg(feature = "procgen")]
pub use building_blocks_procgen as procgen;

#[cfg(feature = "search")]
pub use building_blocks_search as search;

#[cfg(feature = "vox")]
pub use building_blocks_vox as vox;
