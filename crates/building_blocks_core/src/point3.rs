use crate::{
    point::{Bounded, PointOps, SmallOne, SmallZero},
    IntegerPoint, Point, PointN,
};

use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use num::{traits::Pow, Integer, Signed};
use std::cmp::{max, min, Ordering};

/// A 3-dimensional point with scalar type `T`.
pub type Point3<T> = PointN<[T; 3]>;
/// A 3-dimensional point with scalar type `i32`.
pub type Point3i = PointN<[i32; 3]>;

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

impl PointOps for Point3i {
    type Scalar = i32;
}

impl<T> Point for Point3<T>
where
    T: Copy
        + SmallZero
        + SmallOne
        + Signed
        + Add<Output = T>
        + Mul<Output = T>
        + Pow<usize, Output = T>
        + Ord
        + Bounded,
    Point3<T>: PointOps<Scalar = T>,
{
    const ZERO: Self = PointN([T::ZERO; 3]);
    const ONES: Self = PointN([T::ONE; 3]);
    const MIN: Self = PointN([T::MIN; 3]);
    const MAX: Self = PointN([T::MAX; 3]);

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }

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

    fn l1_distance(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs() + diff.z().abs()
    }

    fn l2_distance_squared(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2) + diff.z().pow(2)
    }
}

impl IntegerPoint for Point3i {
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

    fn basis() -> Vec<Self> {
        vec![PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])]
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

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<Point3i> for Point3i {
    type Output = Self;

    fn div(self, rhs: Point3i) -> Self {
        self.vector_div_floor(&rhs)
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

    impl From<Point3i> for na::Point3<f32> {
        fn from(p: Point3i) -> Self {
            na::Point3::new(p.x() as f32, p.y() as f32, p.z() as f32)
        }
    }

    pub fn voxel_containing_point3(p: &na::Point3<f32>) -> Point3i {
        PointN([p.x as i32, p.y as i32, p.z as i32])
    }
}
