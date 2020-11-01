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
pub use extent2::{Extent2, Extent2f, Extent2i};
pub use extent3::{Extent3, Extent3f, Extent3i};
pub use point::{
    Bounded, Distance, DotProduct, IntegerPoint, Norm, NormSquared, Ones, Point, PointN, SmallZero,
};
pub use point2::{pixel_containing_point2f, Point2, Point2f, Point2i};
pub use point3::{voxel_containing_point3f, Point3, Point3f, Point3i};

pub use num;

pub mod prelude {
    pub use super::{
        Bounded, Distance, DotProduct, Extent, Extent2, Extent2f, Extent2i, Extent3, Extent3f,
        Extent3i, ExtentN, IntegerExtent, IntegerPoint, Norm, NormSquared, Ones, Point, Point2,
        Point2f, Point2i, Point3, Point3f, Point3i, PointN, SmallZero,
    };
}
