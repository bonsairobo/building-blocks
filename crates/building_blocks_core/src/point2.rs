use crate::{
    point::SmallOne, Bounded, Distance, DotProduct, IntegerPoint, NormSquared, Ones, Point, PointN,
    SmallZero,
};

use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use num::{traits::Pow, Integer, Signed};
use std::cmp::{max, min, Ordering};

/// A 2-dimensional point with scalar type `T`.
pub type Point2<T> = PointN<[T; 2]>;
/// A 2-dimensional point with scalar type `i32`.
pub type Point2i = PointN<[i32; 2]>;
/// A 2-dimensional point with scalar type `f32`.
pub type Point2f = PointN<[f32; 2]>;

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

    pub fn yx(&self) -> Self {
        PointN([self.y(), self.x()])
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

impl Point2f {
    pub fn round(&self) -> Self {
        self.map_components(|c| c.round())
    }

    pub fn floor(&self) -> Self {
        self.map_components(|c| c.floor())
    }
}

impl<T> Bounded for Point2<T>
where
    T: Bounded,
{
    const MAX: Self = PointN([T::MAX; 2]);
    const MIN: Self = PointN([T::MIN; 2]);
}

impl Point for Point2i {
    type Scalar = i32;

    fn basis() -> Vec<Self> {
        vec![PointN([1, 0]), PointN([0, 1])]
    }

    #[inline]
    fn abs(&self) -> Self {
        PointN([self.x().abs(), self.y().abs()])
    }

    #[inline]
    fn at(&self, component_index: usize) -> Self::Scalar {
        self.0[component_index]
    }

    fn map_components(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self {
        PointN([f(self.x()), f(self.y())])
    }
}

impl Point for Point2f {
    type Scalar = f32;

    fn basis() -> Vec<Self> {
        vec![PointN([1.0, 0.0]), PointN([0.0, 1.0])]
    }

    #[inline]
    fn abs(&self) -> Self {
        PointN([self.x().abs(), self.y().abs()])
    }

    #[inline]
    fn at(&self, component_index: usize) -> Self::Scalar {
        self.0[component_index]
    }

    fn map_components(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self {
        PointN([f(self.x()), f(self.y())])
    }
}

impl<T> SmallZero for Point2<T>
where
    T: SmallZero,
{
    const ZERO: Self = PointN([T::ZERO; 2]);
}

impl<T> Ones for Point2<T>
where
    T: SmallOne,
{
    const ONES: Self = PointN([T::ONE; 2]);
}

impl<T> Distance for Point2<T>
where
    T: Copy + Signed + Add<Output = T> + Pow<u16, Output = T>,
    Point2<T>: Point<Scalar = T>,
{
    fn l1_distance(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs()
    }

    fn l2_distance_squared(&self, other: &Self) -> Self::Scalar {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2)
    }
}

impl NormSquared for Point2i {
    fn norm_squared(&self) -> f32 {
        self.dot(&self) as f32
    }
}

impl NormSquared for Point2f {
    fn norm_squared(&self) -> f32 {
        self.dot(&self)
    }
}

impl<T> DotProduct for Point2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl IntegerPoint for Point2i {
    const MIN: Self = PointN([i32::MIN; 2]);
    const MAX: Self = PointN([i32::MAX; 2]);

    fn join(&self, other: &Self) -> Self {
        PointN([max(self.x(), other.x()), max(self.y(), other.y())])
    }

    fn meet(&self, other: &Self) -> Self {
        PointN([min(self.x(), other.x()), min(self.y(), other.y())])
    }

    #[inline]
    fn left_shift(&self, shift_by: i32) -> Self {
        PointN([self.x() << shift_by, self.y() << shift_by])
    }

    #[inline]
    fn right_shift(&self, shift_by: i32) -> Self {
        PointN([self.x() >> shift_by, self.y() >> shift_by])
    }

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

    fn dimensions_are_powers_of_2(&self) -> bool {
        self.x().is_positive()
            && self.y().is_positive()
            && (self.x() as u32).is_power_of_two()
            && (self.y() as u32).is_power_of_two()
    }

    fn is_cube(&self) -> bool {
        self.x() == self.y()
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

impl Div<f32> for Point2f {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self([self.x() / rhs, self.y() / rhs])
    }
}

impl Div<Self> for Point2f {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self([self.x() / rhs.x(), self.y() / rhs.y()])
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

impl From<Point2i> for Point2f {
    fn from(p: Point2i) -> Self {
        PointN([p.x() as f32, p.y() as f32])
    }
}

impl Point2f {
    pub fn as_2i(&self) -> Point2i {
        PointN([self.x() as i32, self.y() as i32])
    }

    pub fn in_pixel(&self) -> Point2i {
        self.floor().as_2i()
    }
}

#[cfg(feature = "mint")]
pub mod mint_conversions {
    use super::*;

    impl<T> From<mint::Point2<T>> for Point2<T> {
        fn from(p: mint::Point2<T>) -> Self {
            PointN([p.x, p.y])
        }
    }

    impl<T> From<Point2<T>> for mint::Point2<T>
    where
        T: Clone,
    {
        fn from(p: Point2<T>) -> Self {
            mint::Point2::from_slice(&p.0)
        }
    }

    impl<T> From<mint::Vector2<T>> for Point2<T> {
        fn from(p: mint::Vector2<T>) -> Self {
            PointN([p.x, p.y])
        }
    }

    impl<T> From<Point2<T>> for mint::Vector2<T>
    where
        T: Clone,
    {
        fn from(p: Point2<T>) -> Self {
            mint::Vector2::from_slice(&p.0)
        }
    }
}

#[cfg(feature = "nalgebra")]
pub mod nalgebra_conversions {
    use super::*;

    use nalgebra as na;

    impl From<Point2i> for na::Point2<i32> {
        fn from(p: Point2i) -> Self {
            na::Point2::new(p.x(), p.y())
        }
    }
    impl From<Point2f> for na::Point2<f32> {
        fn from(p: Point2f) -> Self {
            na::Point2::new(p.x(), p.y())
        }
    }
    impl From<Point2i> for na::Vector2<i32> {
        fn from(p: Point2i) -> Self {
            na::Vector2::new(p.x(), p.y())
        }
    }
    impl From<Point2f> for na::Vector2<f32> {
        fn from(p: Point2f) -> Self {
            na::Vector2::new(p.x(), p.y())
        }
    }

    impl From<na::Point2<i32>> for Point2i {
        fn from(p: na::Point2<i32>) -> Self {
            PointN([p.x, p.y])
        }
    }
    impl From<na::Point2<f32>> for Point2f {
        fn from(p: na::Point2<f32>) -> Self {
            PointN([p.x, p.y])
        }
    }
    impl From<na::Vector2<i32>> for Point2i {
        fn from(p: na::Vector2<i32>) -> Self {
            PointN([p.x, p.y])
        }
    }
    impl From<na::Vector2<f32>> for Point2f {
        fn from(p: na::Vector2<f32>) -> Self {
            PointN([p.x, p.y])
        }
    }

    impl From<Point2i> for na::Point2<f32> {
        fn from(p: Point2i) -> Self {
            na::Point2::new(p.x() as f32, p.y() as f32)
        }
    }
    impl From<Point2i> for na::Vector2<f32> {
        fn from(p: Point2i) -> Self {
            na::Vector2::new(p.x() as f32, p.y() as f32)
        }
    }
}
