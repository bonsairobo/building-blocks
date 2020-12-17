#![deny(clippy::missing_inline_in_public_items)]

//! The core data types for defining 2D and 3D integer lattices:
//! - `PointN`: an N-dimensional point, most importantly `Point2i` and `Point3i`
//! - `ExtentN`: an N-dimensional extent, most importantly `Extent2i` and `Extent3i`

pub mod axis;
pub mod extent;
pub mod point;

pub use axis::{Axis2, Axis3, Axis3Permutation, SignedAxis2, SignedAxis3};
pub use extent::{
    bounding_extent, Extent2, Extent2f, Extent2i, Extent3, Extent3f, Extent3i, ExtentN,
};
pub use point::{point_traits::*, Point2, Point2f, Point2i, Point3, Point3f, Point3i, PointN};

pub use num;

pub mod prelude {
    pub use super::{
        point::point_traits::*, Axis2, Axis3, Bounded, ComponentwiseIntegerOps, Distance,
        DotProduct, Extent2, Extent2f, Extent2i, Extent3, Extent3f, Extent3i, ExtentN,
        GetComponent, IntegerPoint, MapComponents, Neighborhoods, Norm, NormSquared, Ones, Point,
        Point2, Point2f, Point2i, Point3, Point3f, Point3i, PointN, SmallZero,
    };
}

#[cfg(feature = "glam")]
pub use glam;

#[cfg(feature = "mint")]
pub use mint;

#[cfg(feature = "nalgebra")]
pub use nalgebra as na;
