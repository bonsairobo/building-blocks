use super::*;

use cgmath;

impl From<Point2i> for cgmath::Point2<i32> {
    #[inline]
    /// Converts to cgmath::Point2<i32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let p : Point2i = PointN([1_i32, 2]);
    /// let c = cgmath::Point2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2i) -> Self {
        cgmath::Point2::new(p.x(), p.y())
    }
}

impl From<Point2f> for cgmath::Point2<f32> {
    #[inline]
    /// Converts to cgmath::Point2<f32> from Point2f
    /// ```
    /// # use building_blocks_core::{PointN,Point2f};
    /// let p : Point2f = PointN([1_f32, 2.0]);
    /// let c = cgmath::Point2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2f) -> Self {
        cgmath::Point2::new(p.x(), p.y())
    }
}

impl From<Point2i> for cgmath::Vector2<i32> {
    #[inline]
    /// Converts to cgmath::Vector2<i32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let p : Point2i = PointN([1_i32, 2]);
    /// let c = cgmath::Vector2::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2i) -> Self {
        cgmath::Vector2::new(p.x(), p.y())
    }
}

impl From<Point2f> for cgmath::Vector2<f32> {
    #[inline]
    /// Converts to cgmath::Vector2<f32> from Point2f
    /// ```
    /// # use building_blocks_core::{PointN,Point2f};
    /// let p : Point2f = PointN([1_f32, 2.0]);
    /// let c = cgmath::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: Point2f) -> Self {
        cgmath::Vector2::new(p.x(), p.y())
    }
}

impl From<cgmath::Point2<i32>> for Point2i {
    #[inline]
    /// Converts to Point2i from cgmath::Point2<i32>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let c = cgmath::Point2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Point2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Point2<f32>> for Point2f {
    #[inline]
    /// Converts to Point2f from cgmath::Point2<f32>
    /// ```
    /// # use building_blocks_core::{PointN,Point2f};
    /// let c = cgmath::Point2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Point2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Vector2<i32>> for Point2i {
    #[inline]
    /// Converts to Point2i from cgmath::Vector2<i32>
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let c = cgmath::Vector2::<i32>::new(1,2);
    /// let p = Point2i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Vector2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Vector2<f32>> for Point2f {
    #[inline]
    /// Converts to Point2f from cgmath::Vector2<f32>
    /// ```
    /// # use building_blocks_core::{PointN,Point2f};
    /// let c = cgmath::Vector2::<f32>::new(1.0,2.0);
    /// let p = Point2f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// ```
    fn from(p: cgmath::Vector2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2i> for cgmath::Point2<f32> {
    #[inline]
    /// Converts to cgmath::Point2<f32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    ///
    /// let p : Point2i = PointN([1_i32, 2]);
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
    /// Converts to cgmath::Vector2<f32> from Point2i
    /// ```
    /// # use building_blocks_core::{PointN,Point2i};
    /// let p : Point2i = PointN([1_i32, 2]);
    /// let c = cgmath::Vector2::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// ```
    fn from(p: Point2i) -> Self {
        cgmath::Vector2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point3i> for cgmath::Point3<i32> {
    #[inline]
    /// Converts to cgmath::Point3<i32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1_i32, 2, 3]);
    /// let c = cgmath::Point3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3i) -> Self {
        cgmath::Point3::from(p.0)
    }
}

impl From<Point3f> for cgmath::Point3<f32> {
    #[inline]
    /// Converts to cgmath::Point3<f32> from Point3f
    /// ```
    /// # use building_blocks_core::{PointN,Point3f};
    /// let p : Point3f = PointN([1.0_f32, 2.0, 3.0]);
    /// let c = cgmath::Point3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3f) -> Self {
        cgmath::Point3::from(p.0)
    }
}

impl From<Point3i> for cgmath::Vector3<i32> {
    #[inline]
    /// Converts to cgmath::Vector3<i32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1_i32, 2, 3]);
    /// let c = cgmath::Vector3::<i32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3i) -> Self {
        cgmath::Vector3::from(p.0)
    }
}

impl From<Point3f> for cgmath::Vector3<f32> {
    #[inline]
    /// Converts to cgmath::Vector3<f32> from Point3f
    /// ```
    /// # use building_blocks_core::{PointN,Point3f};
    /// let p : Point3f = PointN([1.0_f32, 2.0, 3.0]);
    /// let c = cgmath::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: Point3f) -> Self {
        cgmath::Vector3::from(p.0)
    }
}

impl From<cgmath::Point3<i32>> for Point3i {
    #[inline]
    /// Converts to Point3i from cgmath::Point3<i32>
    /// ```
    /// # use building_blocks_core::Point3i;
    /// let c = cgmath::Point3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Point3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Point3<f32>> for Point3f {
    #[inline]
    /// Converts to Point3f from cgmath::Point3<f32>
    /// ```
    /// # use building_blocks_core::Point3f;
    ///
    /// let c = cgmath::Point3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Point3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Vector3<i32>> for Point3i {
    #[inline]
    /// Converts to Point3i from cgmath::Vector3<i32>
    /// ```
    /// # use building_blocks_core::Point3i;
    /// let c = cgmath::Vector3::<i32>::new(1,2,3);
    /// let p = Point3i::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Vector3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Vector3<f32>> for Point3f {
    #[inline]
    /// Converts to Point3f from cgmath::Vector3<f32>
    /// ```
    /// # use building_blocks_core::Point3f;
    /// let c = cgmath::Vector3::<f32>::new(1.0,2.0,3.0);
    /// let p = Point3f::from(c);
    /// assert_eq!(c.x , p.x());
    /// assert_eq!(c.y , p.y());
    /// assert_eq!(c.z , p.z());
    /// ```
    fn from(p: cgmath::Vector3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3i> for cgmath::Point3<f32> {
    #[inline]
    /// Converts to cgmath::Point3<f32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1_i32, 2, 3]);
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
    /// Converts to cgmath::Vector3<f32> from Point3i
    /// ```
    /// # use building_blocks_core::{PointN,Point3i};
    /// let p : Point3i = PointN([1_i32, 2, 3]);
    /// let c = cgmath::Vector3::<f32>::from(p);
    /// assert_eq!(c.x , p.x() as f32);
    /// assert_eq!(c.y , p.y() as f32);
    /// assert_eq!(c.z , p.z() as f32);
    /// ```
    fn from(p: Point3i) -> Self {
        cgmath::Vector3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
