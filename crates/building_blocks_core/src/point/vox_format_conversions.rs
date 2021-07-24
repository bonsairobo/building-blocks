//! Provides conversions for point types from the [`vox-format`] crate.
//!
//! [`vox-format`]: https://docs.rs/vox-format

use vox_format::types::{Point, Size, Vector};

use crate::{Extent3i, Point3i, PointN};


impl<T> From<Vector<T>> for PointN<[T; 3]> {
    fn from(v: Vector<T>) -> Self {
        PointN(v.into())
    }
}

impl From<Size> for Extent3i {
    fn from(size: Size) -> Self {
        // Note: This can fail, if the component is greater than `i32::MAX`
        Extent3i::from_min_and_shape(
            Default::default(),
            PointN([size.x as i32, size.y as i32, size.z as i32]),
        )
    }
}

impl From<Point> for Point3i {
    fn from(point: Point) -> Self {
        PointN([point.x as i32, point.y as i32, point.z as i32])
    }
}
