//! The core data types for defining 2D and 3D integer lattices:
//! - `PointN`: an N-dimensional point, most importantly `Point2i` and `Point3i`
//! - `ExtentN`: an N-dimensional extent, most importantly `Extent2i` and `Extent3i`

pub mod extent;
pub mod extent2;
pub mod extent3;
pub mod point;
pub mod point2;
pub mod point3;

pub use extent::{bounding_extent, Extent, ExtentN, IntegerExtent};
pub use extent2::{Extent2, Extent2i};
pub use extent3::{Extent3, Extent3i};
pub use point::{
    Bounded, Distance, DotProduct, IntegerPoint, Norm, Ones, Point, PointN, SmallZero,
};
pub use point2::{Point2, Point2f, Point2i};
pub use point3::{Point3, Point3f, Point3i};

pub use num;

pub mod prelude {
    pub use super::{
        Bounded, Distance, DotProduct, Extent, Extent2, Extent2i, Extent3, Extent3i, ExtentN,
        IntegerExtent, IntegerPoint, Norm, Ones, Point, Point2, Point2f, Point2i, Point3, Point3f,
        Point3i, PointN, SmallZero,
    };
}

#[cfg(feature = "nalg")]
pub use point2::nalgebra_conversions::pixel_containing_point2;

#[cfg(feature = "nalg")]
pub use point3::nalgebra_conversions::voxel_containing_point3;
