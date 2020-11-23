use super::*;

use nalgebra as na;

impl From<Point2i> for na::Point2<i32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        na::Point2::new(p.x(), p.y())
    }
}
impl From<Point2f> for na::Point2<f32> {
    #[inline]
    fn from(p: Point2f) -> Self {
        na::Point2::new(p.x(), p.y())
    }
}
impl From<Point2i> for na::Vector2<i32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        na::Vector2::new(p.x(), p.y())
    }
}
impl From<Point2f> for na::Vector2<f32> {
    #[inline]
    fn from(p: Point2f) -> Self {
        na::Vector2::new(p.x(), p.y())
    }
}

impl From<na::Point2<i32>> for Point2i {
    #[inline]
    fn from(p: na::Point2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}
impl From<na::Point2<f32>> for Point2f {
    #[inline]
    fn from(p: na::Point2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}
impl From<na::Vector2<i32>> for Point2i {
    #[inline]
    fn from(p: na::Vector2<i32>) -> Self {
        PointN([p.x, p.y])
    }
}
impl From<na::Vector2<f32>> for Point2f {
    #[inline]
    fn from(p: na::Vector2<f32>) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2i> for na::Point2<f32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        na::Point2::new(p.x() as f32, p.y() as f32)
    }
}
impl From<Point2i> for na::Vector2<f32> {
    #[inline]
    fn from(p: Point2i) -> Self {
        na::Vector2::new(p.x() as f32, p.y() as f32)
    }
}

impl From<Point3i> for na::Point3<i32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        na::Point3::from(p.0)
    }
}
impl From<Point3f> for na::Point3<f32> {
    #[inline]
    fn from(p: Point3f) -> Self {
        na::Point3::from(p.0)
    }
}
impl From<Point3i> for na::Vector3<i32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        na::Vector3::from(p.0)
    }
}
impl From<Point3f> for na::Vector3<f32> {
    #[inline]
    fn from(p: Point3f) -> Self {
        na::Vector3::from(p.0)
    }
}

impl From<na::Point3<i32>> for Point3i {
    #[inline]
    fn from(p: na::Point3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}
impl From<na::Point3<f32>> for Point3f {
    #[inline]
    fn from(p: na::Point3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}
impl From<na::Vector3<i32>> for Point3i {
    #[inline]
    fn from(p: na::Vector3<i32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}
impl From<na::Vector3<f32>> for Point3f {
    #[inline]
    fn from(p: na::Vector3<f32>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3i> for na::Point3<f32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        na::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
impl From<Point3i> for na::Vector3<f32> {
    #[inline]
    fn from(p: Point3i) -> Self {
        na::Vector3::new(p.x() as f32, p.y() as f32, p.z() as f32)
    }
}
