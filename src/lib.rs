//! Data types, collections, and algorithms for working with maps on 2D and 3D integer lattices.
//! Commonly known as pixel or voxel data.
//!
//! This library is organized into several crates. The most fundamental are:
//! - **core**: lattice point and extent data types
//! - **storage**: storage for lattice maps, i.e. functions defined on `Z^2` and `Z^3`
//!
//! Then you get extra bits of functionality from the others:
//! - **image**: conversion of 2D lattice maps to/from images
//! - **mesh**: 3D mesh generation algorithms
//! - **procgen**: procedural generation of lattice maps
//! - **search**: search algorithms on lattice maps
//! - **vox**: conversion of 3D lattice maps to/from VOX data format
//!
//! To learn the basics about lattice maps, start with these doc pages:
//!
//! - [points](https://docs.rs/building_blocks_core/latest/building_blocks_core/point/struct.PointN.html)
//! - [extents](https://docs.rs/building_blocks_core/latest/building_blocks_core/extent/struct.ExtentN.html)
//! - [arrays](https://docs.rs/building_blocks_storage/latest/building_blocks_storage/array/index.html)
//! - [access traits](https://docs.rs/building_blocks_storage/latest/building_blocks_storage/access/index.html)
//! - [chunk maps](https://docs.rs/building_blocks_storage/latest/building_blocks_storage/chunk_map/index.html)
//! - [transform maps](https://docs.rs/building_blocks_storage/latest/building_blocks_storage/transform_map/index.html)
//! - [fn maps](https://docs.rs/building_blocks_storage/latest/building_blocks_storage/func/index.html)

// TODO: when rust 1.49 is stable, update the the hyperlinks above to use RFC 1946 "intra-links"

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

#[cfg(feature = "procgen")]
pub use building_blocks_procgen as procgen;

#[cfg(feature = "search")]
pub use building_blocks_search as search;

#[cfg(feature = "vox")]
pub use building_blocks_vox as vox;
