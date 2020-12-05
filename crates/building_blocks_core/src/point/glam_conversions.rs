use super::*;

use glam as gl;

impl From<gl::Vec2> for Point2f {
    #[inline]
    fn from(p: gl::Vec2) -> Self {
        PointN([p.x, p.y])
    }
}

impl From<Point2f> for gl::Vec2 {
    #[inline]
    fn from(p: Point2f) -> Self {
        gl::Vec2::new(p.x(), p.y())
    }
}

impl From<gl::Vec3> for Point3f {
    #[inline]
    fn from(p: gl::Vec3) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3f> for gl::Vec3 {
    #[inline]
    fn from(p: Point3f) -> Self {
        gl::Vec3::new(p.x(), p.y(), p.z())
    }
}

impl From<gl::Vec3A> for Point3f {
    #[inline]
    fn from(p: gl::Vec3A) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl From<Point3f> for gl::Vec3A {
    #[inline]
    fn from(p: Point3f) -> Self {
        gl::Vec3A::new(p.x(), p.y(), p.z())
    }
}
