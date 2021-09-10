//! [`cgmath`](https://docs.rs/cgmath) type conversions.
use super::*;

use cgmath;

impl<T: Copy> From<Point2<T>> for cgmath::Point2<T> {
    #[inline]
    /// Converts to `cgmath::Point2<T>` from `Point2<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = cgmath::Point2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let p : Point2f = PointN([1.0, 2.0]);
    /// let c = cgmath::Point2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2<T>) -> Self {
        cgmath::Point2::from(p.0)
    }
}

impl<T: Copy> From<Point2<T>> for cgmath::Vector2<T> {
    #[inline]
    /// Converts to `cgmath::Vector2<T>` from `Point2<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = cgmath::Vector2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let p : Point2f = PointN([1.0, 2.0]);
    /// let c = cgmath::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2<T>) -> Self {
        cgmath::Vector2::from(p.0)
    }
}

impl<T> From<cgmath::Point2<T>> for Point2<T> {
    #[inline]
    /// Converts to `Point2<T>` from cgmath::`Point2<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let c = cgmath::Point2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let c = cgmath::Point2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Point2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl<T> From<cgmath::Vector2<T>> for Point2<T> {
    #[inline]
    /// Converts to `Point2<T>` from `cgmath::Vector2<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i,Point2f};
    /// let c = cgmath::Vector2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// let c = cgmath::Vector2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Vector2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2i> for cgmath::Point2<f32> {
    #[inline]
    /// Converts to `cgmath::Point2<f32>` from `Point2i`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    ///
    /// let p : Point2i = PointN([1, 2]);
    /// let c = cgmath::Point2::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// ```
    fn from(p: Point2i) -> Self {
        cgmath::Point2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point2i> for cgmath::Vector2<f32> {
    #[inline]
    /// Converts to `cgmath::Vector2<f32>` from `Point2i`
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let p : Point2i = PointN([1, 2]);
    /// let c = cgmath::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// ```
    fn from(p: Point2i) -> Self {
        cgmath::Vector2::new(p.x() as f32, p.y() as f32)
    }
}

impl<T: Copy> From<Point3<T>> for cgmath::Point3<T> {
    #[inline]
    /// Converts to `cgmath::Point3<T>` from `Point3<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point3i,Point3f};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = cgmath::Vector3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let p : Point3f = PointN([1.0, 2.0, 3.0]);
    /// let c = cgmath::Point3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3<T>) -> Self {
        cgmath::Point3::from(p.0)
    }
}

impl<T: Copy> From<Point3<T>> for cgmath::Vector3<T> {
    #[inline]
    /// Converts to `cgmath::Vector3<T>` from `Point3<T>`
    /// ```
    /// # use building_blocks_core::{PointN,Point3i,Point3f};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = cgmath::Vector3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let p : Point3f = PointN([1.0, 2.0, 3.0]);
    /// let c = cgmath::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3<T>) -> Self {
        cgmath::Vector3::from(p.0)
    }
}

impl<T> From<cgmath::Point3<T>> for Point3<T> {
    #[inline]
    /// Converts to `Point3<T>` from `cgmath::Point3<T>`
    /// ```
    /// # use building_blocks_core::{Point3i,Point3f};
    /// let c = cgmath::Point3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let c = cgmath::Point3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Point3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl<T> From<cgmath::Vector3<T>> for Point3<T> {
    #[inline]
    /// Converts to `Point3<T>` from `cgmath::Vector3<T>`
    /// ```
    /// # use building_blocks_core::{Point3i,Point3f};
    /// let c = cgmath::Vector3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// let c = cgmath::Vector3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Vector3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3i> for cgmath::Point3<f32> {
    #[inline]
    /// Converts to `cgmath::Point3<f32>` from `Point3i`
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = cgmath::Point3::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// assert_eq!(c.z , p.z() as f32);
    /// ```
    fn from(p: Point3i) -> Self {
        cgmath::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}

impl From<Point3i> for cgmath::Vector3<f32> {
    #[inline]
    /// Converts to `cgmath::Vector3<f32>` from `Point3i`
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1, 2, 3]);
    /// let c = cgmath::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// assert_eq!(c.z , p.z() as f32);
    /// ```
    fn from(p: Point3i) -> Self {
        cgmath::Vector3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
