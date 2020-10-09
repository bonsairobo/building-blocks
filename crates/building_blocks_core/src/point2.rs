use crate::{
    point::{Bounded, PointOps, SmallOne, SmallZero},
    IntegerPoint, Point, PointN,
};

use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use num::{traits::Pow, Integer, Signed};
use std::cmp::{max, min, Ordering};

/// A 2-dimensional point with scalar type `T`.
pub type Point2<T> = PointN<[T; 2]>;
/// A 2-dimensional point with scalar type `i32`.
pub type Point2i = PointN<[i32; 2]>;

impl<T> Point2<T> {
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }
}

impl<T> Point2<T>
where
    T: Copy,
{
    pub fn x(&self) -> T {
        self.0[0]
    }

    pub fn y(&self) -> T {
        self.0[1]
    }
}

impl<T> Point2<T>
where
    T: Copy + Integer,
{
    pub fn vector_div_floor(&self, rhs: &Self) -> Self {
        PointN([self.x().div_floor(&rhs.x()), self.y().div_floor(&rhs.y())])
    }

    pub fn scalar_div_floor(&self, rhs: T) -> Self {
        PointN([self.x().div_floor(&rhs), self.y().div_floor(&rhs)])
    }
}

impl PointOps for Point2i {
    type Scalar = i32;
}

impl<T> Point for Point2<T>
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
    Point2<T>: PointOps<Scalar = T>,
{
    const ZERO: Self = PointN([T::ZERO; 2]);
    const ONES: Self = PointN([T::ONE; 2]);
    const MIN: Self = PointN([T::MIN; 2]);
    const MAX: Self = PointN([T::MAX; 2]);

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y()
    }

    fn join(&self, other: &Self) -> Self {
        PointN([max(self.x(), other.x()), max(self.y(), other.y())])
    }

    fn meet(&self, other: &Self) -> Self {
        PointN([min(self.x(), other.x()), min(self.y(), other.y())])
    }

    fn l1_distance(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs()
    }

    fn l2_distance_squared(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2)
    }
}

impl IntegerPoint for Point2i {
    fn corner_offsets() -> Vec<Self> {
        vec![
            PointN([0, 0]),
            PointN([1, 0]),
            PointN([0, 1]),
            PointN([1, 1]),
        ]
    }

    fn von_neumann_offsets() -> Vec<Self> {
        vec![
            PointN([-1, 0]),
            PointN([1, 0]),
            PointN([0, -1]),
            PointN([0, 1]),
        ]
    }

    fn moore_offsets() -> Vec<Self> {
        vec![
            PointN([-1, -1]),
            PointN([0, -1]),
            PointN([1, -1]),
            PointN([-1, 0]),
            PointN([1, 0]),
            PointN([-1, 1]),
            PointN([0, 1]),
            PointN([1, 1]),
        ]
    }
}

impl<T> Add for Point2<T>
where
    T: Copy + AddAssign,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut sum = self;
        *sum.x_mut() += rhs.x();
        *sum.y_mut() += rhs.y();

        sum
    }
}

impl<T> Sub for Point2<T>
where
    T: Copy + SubAssign,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut sub = self;
        *sub.x_mut() -= rhs.x();
        *sub.y_mut() -= rhs.y();

        sub
    }
}

// This particular partial order allows us to say that an `Extent2i` e contains a `Point2i` p iff p
// is GEQ the minimum of e and p is LEQ the maximum of e.
impl<T> PartialOrd for Point2<T>
where
    T: Copy + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self < other {
            Some(Ordering::Less)
        } else if self > other {
            Some(Ordering::Greater)
        } else if self.x() == other.x() && self.y() == other.y() {
            Some(Ordering::Equal)
        } else {
            None
        }
    }

    fn lt(&self, other: &Self) -> bool {
        self.x() < other.x() && self.y() < other.y()
    }

    fn gt(&self, other: &Self) -> bool {
        self.x() > other.x() && self.y() > other.y()
    }

    fn le(&self, other: &Self) -> bool {
        self.x() <= other.x() && self.y() <= other.y()
    }

    fn ge(&self, other: &Self) -> bool {
        self.x() >= other.x() && self.y() >= other.y()
    }
}

impl<T> Mul<T> for Point2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        PointN([rhs * self.x(), rhs * self.y()])
    }
}

impl<T> Mul<Point2<T>> for Point2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        PointN([other.x() * self.x(), other.y() * self.y()])
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<i32> for Point2i {
    type Output = Self;

    fn div(self, rhs: i32) -> Self {
        self.scalar_div_floor(rhs)
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<Point2i> for Point2i {
    type Output = Self;

    fn div(self, rhs: Point2i) -> Self {
        self.vector_div_floor(&rhs)
    }
}
