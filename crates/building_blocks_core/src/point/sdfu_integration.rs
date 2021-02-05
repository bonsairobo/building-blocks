use super::Point3f;
use super::*;

use mt::MaxMin;
use sdfu::mathtypes as mt;

impl mt::Zero for Point2f {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
}

impl mt::One for Point2f {
    #[inline]
    fn one() -> Self {
        Self::ONES
    }
}

impl mt::Clamp for Point2f {
    #[inline]
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.min(high).max(low)
    }
}

impl mt::MaxMin for Point2f {
    #[inline]
    fn max(&self, other: Self) -> Self {
        self.join(&other)
    }
    #[inline]
    fn min(&self, other: Self) -> Self {
        self.meet(&other)
    }
}

impl mt::Zero for Point3f {
    #[inline]
    fn zero() -> Self {
        Point3f::ZERO
    }
}

impl mt::One for Point3f {
    #[inline]
    fn one() -> Self {
        Point3f::ONES
    }
}

impl mt::Clamp for Point3f {
    #[inline]
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.min(high).max(low)
    }
}

impl mt::MaxMin for Point3f {
    #[inline]
    fn max(&self, other: Self) -> Self {
        self.join(&other)
    }
    #[inline]
    fn min(&self, other: Self) -> Self {
        self.meet(&other)
    }
}

impl mt::Vec2<f32> for Point2f {
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Self([x, y])
    }
    #[inline]
    fn x(&self) -> f32 {
        self.x()
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y()
    }
}

impl mt::Vec3<f32> for Point3f {
    #[inline]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }
    #[inline]
    fn x(&self) -> f32 {
        self.x()
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y()
    }
    #[inline]
    fn z(&self) -> f32 {
        self.z()
    }
}

impl mt::Vec<f32> for Point2f {
    type Dimension = mt::Dim3D;
    type Vec2 = Point2f;
    type Vec3 = Point3f;

    #[inline]
    fn dot(&self, other: Self) -> f32 {
        DotProduct::dot(self, &other)
    }

    #[inline]
    fn abs(&self) -> Self {
        Abs::abs(self)
    }

    #[inline]
    fn normalized(&self) -> Self {
        *self / self.norm()
    }

    #[inline]
    fn magnitude(&self) -> f32 {
        self.norm()
    }
}

impl mt::Vec<f32> for Point3f {
    type Dimension = mt::Dim3D;
    type Vec2 = Point2f;
    type Vec3 = Point3f;

    #[inline]
    fn dot(&self, other: Self) -> f32 {
        DotProduct::dot(self, &other)
    }

    #[inline]
    fn abs(&self) -> Self {
        Abs::abs(self)
    }

    #[inline]
    fn normalized(&self) -> Self {
        *self / self.norm()
    }

    #[inline]
    fn magnitude(&self) -> f32 {
        self.norm()
    }
}
