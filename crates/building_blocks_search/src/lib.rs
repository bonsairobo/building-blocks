#![allow(
    clippy::type_complexity,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    nonstandard_style,
    unused,
    future_incompatible
)]
#![warn(clippy::doc_markdown)]

mod find_surface;
mod flood_fill;
mod grid_ray_traversal;
mod pathfinding;

pub use self::pathfinding::*;
pub use find_surface::*;
pub use flood_fill::*;
pub use grid_ray_traversal::*;

#[cfg(feature = "ncollide")]
pub mod collision;
#[cfg(feature = "ncollide")]
pub use collision::*;

#[cfg(feature = "ncollide")]
pub mod octree_dbvt;
#[cfg(feature = "ncollide")]
pub use octree_dbvt::*;

#[cfg(feature = "ncollide")]
pub use ncollide3d;
