#![deny(clippy::missing_inline_in_public_items)]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    nonstandard_style,
    unused,
    future_incompatible
)]
#![warn(clippy::doc_markdown)]
#![doc = include_str!("crate_doc.md")]

mod axis;
mod extent;
mod morton;
mod orthant;
mod point;
mod sphere;

pub use axis::*;
pub use extent::*;
pub use morton::*;
pub use orthant::*;
pub use point::*;
pub use sphere::*;

pub use bytemuck;
pub use num;
pub use itertools;

#[doc(hidden)]
pub mod prelude {
    pub use super::{
        point::point_traits::*, Axis2, Axis3, Extent2, Extent2f, Extent2i, Extent3, Extent3f,
        Extent3i, ExtentN, Morton2, Morton3, Octant, Orthant, Point2, Point2f, Point2i, Point3,
        Point3f, Point3i, PointN, Quadrant, Sphere2, Sphere3,
    };
}

#[cfg(feature = "glam")]
pub use glam;

#[cfg(feature = "mint")]
pub use mint;

#[cfg(feature = "nalgebra")]
pub use nalgebra as na;

#[cfg(feature = "sdfu")]
pub use sdfu;

/// Given an array of 4 corners of a rectangle, this contains pairs of indices that make up the edges.
pub const EDGES_2: [[usize; 2]; 4] = [[0b00, 0b01], [0b00, 0b10], [0b10, 0b11], [0b01, 0b11]];

/// Given an array of 8 corners of a rectangular prism, this contains pairs of indices that make up the edges.
pub const EDGES_3: [[usize; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b011, 0b111],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b101, 0b111],
    [0b110, 0b111],
];
