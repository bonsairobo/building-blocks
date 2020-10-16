use crate::{
    point::SmallOne, Bounded, Distance, DotProduct, IntegerPoint, Norm, Ones, Point, Point2,
    PointN, SmallZero,
};

use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use num::{traits::Pow, Integer, Signed};
use std::cmp::{max, min, Ordering};

/// A 3-dimensional point with scalar type `T`.
pub type Point3<T> = PointN<[T; 3]>;
/// A 3-dimensional point with scalar type `i32`.
pub type Point3i = PointN<[i32; 3]>;
/// A 3-dimensional point with scalar type `f32`.
pub type Point3f = PointN<[f32; 3]>;

impl<T> PointN<[T; 3]> {
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }

    pub fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }
}

impl<T> PointN<[T; 3]>
where
    T: Copy,
{
    pub fn x(&self) -> T {
        self.0[0]
    }

    pub fn y(&self) -> T {
        self.0[1]
    }

    pub fn z(&self) -> T {
        self.0[2]
    }

    pub fn xy(&self) -> Point2<T> {
        PointN([self.x(), self.y()])
    }

    pub fn yx(&self) -> Point2<T> {
        PointN([self.y(), self.x()])
    }

    pub fn yz(&self) -> Point2<T> {
        PointN([self.y(), self.z()])
    }

    pub fn zy(&self) -> Point2<T> {
        PointN([self.z(), self.y()])
    }

    pub fn zx(&self) -> Point2<T> {
        PointN([self.z(), self.x()])
    }

    pub fn xz(&self) -> Point2<T> {
        PointN([self.x(), self.z()])
    }

    pub fn yzx(&self) -> Point3<T> {
        PointN([self.y(), self.z(), self.x()])
    }

    pub fn zxy(&self) -> Point3<T> {
        PointN([self.z(), self.x(), self.y()])
    }

    pub fn zyx(&self) -> Point3<T> {
        PointN([self.z(), self.y(), self.x()])
    }
}

impl<T> Point3<T>
where
    T: Copy + Integer,
{
    pub fn vector_div_floor(&self, rhs: &Self) -> Self {
        PointN([
            self.x().div_floor(&rhs.x()),
            self.y().div_floor(&rhs.y()),
            self.z().div_floor(&rhs.z()),
        ])
    }

    pub fn scalar_div_floor(&self, rhs: T) -> Self {
        PointN([
            self.x().div_floor(&rhs),
            self.y().div_floor(&rhs),
            self.z().div_floor(&rhs),
        ])
    }
}

impl<T> Bounded for Point3<T>
where
    T: Bounded,
{
    const MAX: Self = PointN([T::MAX; 3]);
    const MIN: Self = PointN([T::MIN; 3]);
}

impl Point for Point3i {
    type Scalar = i32;

    fn basis() -> Vec<Self> {
        vec![PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])]
    }
}

impl Point for Point3f {
    type Scalar = f32;

    fn basis() -> Vec<Self> {
        vec![
            PointN([1.0, 0.0, 0.0]),
            PointN([0.0, 1.0, 0.0]),
            PointN([0.0, 0.0, 1.0]),
        ]
    }
}

impl<T> SmallZero for Point3<T>
where
    T: SmallZero,
{
    const ZERO: Self = PointN([T::ZERO; 3]);
}

impl<T> Ones for Point3<T>
where
    T: SmallOne,
{
    const ONES: Self = PointN([T::ONE; 3]);
}

impl<T> Distance for Point3<T>
where
    T: Copy + Signed + Add<Output = T> + Pow<i32, Output = T>,
    Point3<T>: Point<Scalar = T>,
{
    fn l1_distance(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs() + diff.z().abs()
    }

    fn l2_distance_squared(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2) + diff.z().pow(2)
    }
}

impl Norm for Point3i {
    fn norm(&self) -> f32 {
        (self.dot(&self) as f32).sqrt()
    }
}

impl Norm for Point3f {
    fn norm(&self) -> f32 {
        self.dot(&self).sqrt()
    }
}

impl<T> DotProduct for Point3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl IntegerPoint for Point3i {
    const MIN: Self = PointN([i32::MIN; 3]);
    const MAX: Self = PointN([i32::MAX; 3]);

    fn join(&self, other: &Self) -> Self {
        PointN([
            max(self.x(), other.x()),
            max(self.y(), other.y()),
            max(self.z(), other.z()),
        ])
    }

    fn meet(&self, other: &Self) -> Self {
        PointN([
            min(self.x(), other.x()),
            min(self.y(), other.y()),
            min(self.z(), other.z()),
        ])
    }

    #[inline]
    fn left_shift(&self, shift_by: i32) -> Self {
        PointN([
            self.x() << shift_by,
            self.y() << shift_by,
            self.z() << shift_by,
        ])
    }

    #[inline]
    fn right_shift(&self, shift_by: i32) -> Self {
        PointN([
            self.x() >> shift_by,
            self.y() >> shift_by,
            self.z() >> shift_by,
        ])
    }

    fn corner_offsets() -> Vec<Self> {
        vec![
            PointN([0, 0, 0]),
            PointN([1, 0, 0]),
            PointN([0, 1, 0]),
            PointN([1, 1, 0]),
            PointN([0, 0, 1]),
            PointN([1, 0, 1]),
            PointN([0, 1, 1]),
            PointN([1, 1, 1]),
        ]
    }

    fn von_neumann_offsets() -> Vec<Self> {
        vec![
            PointN([-1, 0, 0]),
            PointN([1, 0, 0]),
            PointN([0, -1, 0]),
            PointN([0, 1, 0]),
            PointN([0, 0, -1]),
            PointN([0, 0, 1]),
        ]
    }

    // Because there are "moore" of them... huehuehue
    fn moore_offsets() -> Vec<Self> {
        vec![
            PointN([-1, -1, -1]),
            PointN([0, -1, -1]),
            PointN([1, -1, -1]),
            PointN([-1, 0, -1]),
            PointN([0, 0, -1]),
            PointN([1, 0, -1]),
            PointN([-1, 1, -1]),
            PointN([0, 1, -1]),
            PointN([1, 1, -1]),
            PointN([-1, -1, 0]),
            PointN([0, -1, 0]),
            PointN([1, -1, 0]),
            PointN([-1, 0, 0]),
            PointN([1, 0, 0]),
            PointN([-1, 1, 0]),
            PointN([0, 1, 0]),
            PointN([1, 1, 0]),
            PointN([-1, -1, 1]),
            PointN([0, -1, 1]),
            PointN([1, -1, 1]),
            PointN([-1, 0, 1]),
            PointN([0, 0, 1]),
            PointN([1, 0, 1]),
            PointN([-1, 1, 1]),
            PointN([0, 1, 1]),
            PointN([1, 1, 1]),
        ]
    }
}

impl<T> Add for PointN<[T; 3]>
where
    T: AddAssign + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut sum = self;
        *sum.x_mut() += rhs.x();
        *sum.y_mut() += rhs.y();
        *sum.z_mut() += rhs.z();

        sum
    }
}

impl<T> Sub for PointN<[T; 3]>
where
    T: SubAssign + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut sub = self;
        *sub.x_mut() -= rhs.x();
        *sub.y_mut() -= rhs.y();
        *sub.z_mut() -= rhs.z();

        sub
    }
}

// This particular partial order allows us to say that an `Extent3i` e contains a `Point3i` p iff p
// is GEQ the minimum of e and p is LEQ the maximum of e.
impl<T> PartialOrd for Point3<T>
where
    T: Copy + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self < other {
            Some(Ordering::Less)
        } else if self > other {
            Some(Ordering::Greater)
        } else if self.x() == other.x() && self.y() == other.y() && self.z() == other.z() {
            Some(Ordering::Equal)
        } else {
            None
        }
    }

    fn lt(&self, other: &Self) -> bool {
        self.x() < other.x() && self.y() < other.y() && self.z() < other.z()
    }

    fn gt(&self, other: &Self) -> bool {
        self.x() > other.x() && self.y() > other.y() && self.z() > other.z()
    }

    fn le(&self, other: &Self) -> bool {
        self.x() <= other.x() && self.y() <= other.y() && self.z() <= other.z()
    }

    fn ge(&self, other: &Self) -> bool {
        self.x() >= other.x() && self.y() >= other.y() && self.z() >= other.z()
    }
}

impl<T> Mul<T> for Point3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        PointN([rhs * self.x(), rhs * self.y(), rhs * self.z()])
    }
}

impl<T> Mul<Point3<T>> for Point3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        PointN([
            other.x() * self.x(),
            other.y() * self.y(),
            other.z() * self.z(),
        ])
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<i32> for Point3i {
    type Output = Self;

    fn div(self, rhs: i32) -> Self {
        self.scalar_div_floor(rhs)
    }
}

impl Div<f32> for Point3f {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self([self.x() / rhs, self.y() / rhs, self.z() / rhs])
    }
}

impl Div<Self> for Point3f {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self([self.x() / rhs.x(), self.y() / rhs.y(), self.z() / rhs.z()])
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<Point3i> for Point3i {
    type Output = Self;

    fn div(self, rhs: Point3i) -> Self {
        self.vector_div_floor(&rhs)
    }
}

impl From<Point3i> for Point3f {
    fn from(p: Point3i) -> Self {
        PointN([p.x() as f32, p.y() as f32, p.z() as f32])
    }
}

#[cfg(feature = "nalg")]
pub mod nalgebra_conversions {
    use super::*;

    use nalgebra as na;

    impl From<Point3i> for na::Point3<i32> {
        fn from(p: Point3i) -> Self {
            na::Point3::new(p.x(), p.y(), p.z())
        }
    }
    impl From<Point3f> for na::Point3<f32> {
        fn from(p: Point3f) -> Self {
            na::Point3::new(p.x(), p.y(), p.z())
        }
    }

    impl From<na::Point3<i32>> for Point3i {
        fn from(p: na::Point3<i32>) -> Self {
            PointN([p.x, p.y, p.z])
        }
    }
    impl From<na::Point3<f32>> for Point3f {
        fn from(p: na::Point3<f32>) -> Self {
            PointN([p.x, p.y, p.z])
        }
    }

    impl From<Point3i> for na::Point3<f32> {
        fn from(p: Point3i) -> Self {
            na::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
        }
    }

    pub fn voxel_containing_point3(p: &na::Point3<f32>) -> Point3i {
        PointN([p.x as i32, p.y as i32, p.z as i32])
    }
}
