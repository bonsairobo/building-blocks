#![deny(clippy::missing_inline_in_public_items)]

//! The core data types for defining 2D and 3D integer lattices:
//! - `PointN`: an N-dimensional point, most importantly `Point2i` and `Point3i`
//! - `ExtentN`: an N-dimensional extent, most importantly `Extent2i` and `Extent3i`

pub mod axis;
pub mod extent;
pub mod morton;
pub mod orthant;
pub mod point;

pub use axis::{Axis2, Axis3, Axis3Permutation, SignedAxis2, SignedAxis3};
pub use extent::{
    bounding_extent, Extent2, Extent2f, Extent2i, Extent3, Extent3f, Extent3i, ExtentN,
};
pub use morton::*;
pub use orthant::*;
pub use point::{point_traits::*, Point2, Point2f, Point2i, Point3, Point3f, Point3i, PointN};

pub use bytemuck;
pub use num;

pub mod prelude {
    pub use super::{
        point::point_traits::*, Axis2, Axis3, Bounded, ConstZero, Distance, DotProduct, Extent2,
        Extent2f, Extent2i, Extent3, Extent3f, Extent3i, ExtentN, GetComponent, IntegerPoint,
        MapComponents, Morton2, Morton3, Neighborhoods, Norm, Octant, Ones, Orthant, Point, Point2,
        Point2f, Point2i, Point3, Point3f, Point3i, PointN, Quadrant,
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
