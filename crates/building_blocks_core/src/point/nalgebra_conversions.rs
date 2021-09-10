//! [`nalgebra`](https://nalgebra.org) type conversions.
use super::*;

use nalgebra as na;
use nalgebra::base::Scalar;

impl<T: Scalar> From<Point2<T>> for na::Point2<T> {
    #[inline]
    /// Converts to nalgebra::Point2<T> from Point2<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = nalgebra::Point2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let p : Point2f = PointN([1.0, 2.0]);
    /// let c = nalgebra::Point2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2<T>) -> Self {
        na::Point2::from(p.0)
    }
}

impl<T: Scalar> From<Point2<T>> for nalgebra::Vector2<T> {
    #[inline]
    /// Converts to nalgebra::Vector2<T> from Point2<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = nalgebra::Vector2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let p : Point2f = PointN([1.0, 2.0]);
    /// let c = nalgebra::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2<T>) -> Self {
        na::Vector2::from(p.0)
    }
}

impl<T: Copy + Scalar> From<na::Point2<T>> for Point2<T> {
    #[inline]
    /// Converts to Point2<T> from nalgebra::Point2<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let c = nalgebra::Point2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let c = nalgebra::Point2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: na::Point2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl<T: Copy + Scalar> From<na::Vector2<T>> for Point2<T> {
    #[inline]
    /// Converts to Point2<T> from nalgebra::Vector2<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let c = nalgebra::Vector2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let c = nalgebra::Vector2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: na::Vector2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2i> for na::Point2<f32> {
    #[inline]
    /// Converts to nalgebra::Point2<f32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    ///
    /// let p : Point2i = PointN([1, 2]);
    /// let c = nalgebra::Point2::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// ```
    fn from(p: Point2i) -> Self {
        na::Point2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point2i> for na::Vector2<f32> {
    #[inline]
    /// Converts to nalgebra::Vector2<f32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = nalgebra::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// ```
    fn from(p: Point2i) -> Self {
        na::Vector2::new(p.x() as f32, p.y() as f32)
    }
}

impl<T: Scalar> From<Point3<T>> for na::Point3<T> {
    #[inline]
    /// Converts to nalgebra::Point3<T> from Point3<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point3i,Point3f};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = nalgebra::Vector3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let p : Point3f = PointN([1.0, 2.0, 3.0]);
    /// let c = nalgebra::Point3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3<T>) -> Self {
        na::Point3::from(p.0)
    }
}

impl<T: Scalar> From<Point3<T>> for na::Vector3<T> {
    #[inline]
    /// Converts to nalgebra::Vector3<T> from Point3<T>
    /// ```
    /// # use building_blocks_core::{PointN,Point3i,Point3f};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = nalgebra::Vector3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let p : Point3f = PointN([1.0, 2.0, 3.0]);
    /// let c = nalgebra::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3<T>) -> Self {
        na::Vector3::from(p.0)
    }
}

impl<T: Copy + Scalar> From<na::Point3<T>> for Point3<T> {
    #[inline]
    /// Converts to Point3<T> from nalgebra::Point3<T>
    /// ```
    /// # use building_blocks_core::{Point3i,Point3f};
    /// let c = nalgebra::Point3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let c = nalgebra::Point3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: na::Point3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl<T: Copy + Scalar> From<na::Vector3<T>> for Point3<T> {
    #[inline]
    /// Converts to Point3<T> from nalgebra::Vector3<T>
    /// ```
    /// # use building_blocks_core::{Point3i,Point3f};
    /// let c = nalgebra::Vector3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let c = nalgebra::Vector3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: na::Vector3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3i> for na::Point3<f32> {
    #[inline]
    /// Converts to nalgebra::Point3<f32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = nalgebra::Point3::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// assert_eq!(c.z , p.z() as f32);
    /// ```
    fn from(p: Point3i) -> Self {
        na::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}

impl From<Point3i> for na::Vector3<f32> {
    #[inline]
    /// Converts to nalgebra::Vector3<f32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = nalgebra::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// assert_eq!(c.z , p.z() as f32);
    /// ```
    fn from(p: Point3i) -> Self {
        na::Vector3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
