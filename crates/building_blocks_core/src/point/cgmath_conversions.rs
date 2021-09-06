use super::*;

use cgmath;

impl From<Point2i> for cgmath::Point2<i32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        cgmath::Point2::new(p.x(), p.y())
    }
}

impl From<Point2f> for cgmath::Point2<f32> {
    #[inline]
    fn from(p: Point2f) -> Self {
        cgmath::Point2::new(p.x(), p.y())
    }
}

impl From<Point2i> for cgmath::Vector2<i32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        cgmath::Vector2::new(p.x(), p.y())
    }
}

impl From<Point2f> for cgmath::Vector2<f32> {
    #[inline]
    fn from(p: Point2f) -> Self {
        cgmath::Vector2::new(p.x(), p.y())
    }
}

impl From<cgmath::Point2<i32>> for Point2i {
    #[inline]
    fn from(p: cgmath::Point2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Point2<f32>> for Point2f {
    #[inline]
    fn from(p: cgmath::Point2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Vector2<i32>> for Point2i {
    #[inline]
    fn from(p: cgmath::Vector2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<cgmath::Vector2<f32>> for Point2f {
    #[inline]
    fn from(p: cgmath::Vector2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2i> for cgmath::Point2<f32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        cgmath::Point2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point2i> for cgmath::Vector2<f32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        cgmath::Vector2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point3i> for cgmath::Point3<i32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        cgmath::Point3::from(p.0)
    }
}
impl From<Point3f> for cgmath::Point3<f32> {
    #[inline]
    fn from(p: Point3f) -> Self {
        cgmath::Point3::from(p.0)
    }
}
impl From<Point3i> for cgmath::Vector3<i32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        cgmath::Vector3::from(p.0)
    }
}
impl From<Point3f> for cgmath::Vector3<f32> {
    #[inline]
    fn from(p: Point3f) -> Self {
        cgmath::Vector3::from(p.0)
    }
}

impl From<cgmath::Point3<i32>> for Point3i {
    #[inline]
    fn from(p: cgmath::Point3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Point3<f32>> for Point3f {
    #[inline]
    fn from(p: cgmath::Point3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Vector3<i32>> for Point3i {
    #[inline]
    fn from(p: cgmath::Vector3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<cgmath::Vector3<f32>> for Point3f {
    #[inline]
    fn from(p: cgmath::Vector3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3i> for cgmath::Point3<f32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        cgmath::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}

impl From<Point3i> for cgmath::Vector3<f32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        cgmath::Vector3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
