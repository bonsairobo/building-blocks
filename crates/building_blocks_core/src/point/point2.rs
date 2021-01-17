use crate::{Axis2, IterExtent};

use super::{point_traits::*, PointN};

use core::ops::{Add, Div, Mul, Range};
use itertools::{iproduct, Product};
use num::{traits::Pow, Integer, Signed};
use std::cmp::Ordering;

/// A 2-dimensional point with scalar type `T`.
pub type Point2<T> = PointN<[T; 2]>;
/// A 2-dimensional point with scalar type `i32`.
pub type Point2i = PointN<[i32; 2]>;
/// A 2-dimensional point with scalar type `f32`.
pub type Point2f = PointN<[f32; 2]>;

impl<T> Point2<T> {
    #[inline]
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    #[inline]
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }

    #[inline]
    pub fn axis_component(&self, axis: Axis2) -> &T {
        &self.0[axis.index()]
    }

    #[inline]
    pub fn axis_component_mut(&mut self, axis: Axis2) -> &mut T {
        &mut self.0[axis.index()]
    }
}

impl<T> Point2<T>
where
    T: Copy,
{
    #[inline]
    pub fn x(&self) -> T {
        self.0[0]
    }

    #[inline]
    pub fn y(&self) -> T {
        self.0[1]
    }

    #[inline]
    pub fn yx(&self) -> Self {
        PointN([self.y(), self.x()])
    }
}

impl Point2f {
    #[inline]
    pub fn round(&self) -> Self {
        self.map_components_unary(|c| c.round())
    }

    #[inline]
    pub fn floor(&self) -> Self {
        self.map_components_unary(|c| c.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Point2f {
        self.map_components_unary(|c| c.ceil())
    }

    #[inline]
    pub fn fract(&self) -> Point2f {
        self.map_components_unary(|c| c.fract())
    }

    #[inline]
    pub fn as_2i(&self) -> Point2i {
        PointN([self.x() as i32, self.y() as i32])
    }

    #[inline]
    pub fn in_pixel(&self) -> Point2i {
        self.floor().as_2i()
    }
}

impl<T> Bounded for Point2<T>
where
    T: Bounded,
{
    const MAX: Self = PointN([T::MAX; 2]);
    const MIN: Self = PointN([T::MIN; 2]);
}

impl<T> MapComponents for Point2<T>
where
    T: Copy,
{
    type Scalar = T;

    #[inline]
    fn map_components_unary(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self {
        PointN([f(self.x()), f(self.y())])
    }

    #[inline]
    fn map_components_binary(
        &self,
        other: &Self,
        f: impl Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self {
        PointN([f(self.x(), other.x()), f(self.y(), other.y())])
    }
}

impl<T> GetComponent for Point2<T>
where
    T: Copy,
{
    type Scalar = T;

    #[inline]
    fn at(&self, component_index: usize) -> T {
        self.0[component_index]
    }
}

impl Point for Point2i {
    type Scalar = i32;

    #[inline]
    fn basis() -> Vec<Self> {
        vec![PointN([1, 0]), PointN([0, 1])]
    }

    #[inline]
    fn volume(&self) -> <Self as Point>::Scalar {
        self.x() * self.y()
    }
}

impl Point for Point2f {
    type Scalar = f32;

    #[inline]
    fn basis() -> Vec<Self> {
        vec![PointN([1.0, 0.0]), PointN([0.0, 1.0])]
    }

    #[inline]
    fn volume(&self) -> <Self as Point>::Scalar {
        self.x() * self.y()
    }
}

impl<T> ConstZero for Point2<T>
where
    T: ConstZero,
{
    const ZERO: Self = PointN([T::ZERO; 2]);
}

impl<T> Ones for Point2<T>
where
    T: ConstOne,
{
    const ONES: Self = PointN([T::ONE; 2]);
}

impl<T> Distance for Point2<T>
where
    T: Copy + Signed + Add<Output = T> + Pow<u16, Output = T>,
    Point2<T>: Point<Scalar = T>,
{
    #[inline]
    fn l1_distance(&self, other: &Self) -> T {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs()
    }

    #[inline]
    fn l2_distance_squared(&self, other: &Self) -> T {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2)
    }
}

impl NormSquared for Point2i {
    #[inline]
    fn norm_squared(&self) -> f32 {
        self.dot(&self) as f32
    }
}

impl NormSquared for Point2f {
    #[inline]
    fn norm_squared(&self) -> f32 {
        self.dot(&self)
    }
}

impl<T> DotProduct for Point2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl IntegerPoint<[i32; 2]> for Point2i {
    #[inline]
    fn dimensions_are_powers_of_2(&self) -> bool {
        self.x().is_positive()
            && self.y().is_positive()
            && (self.x() as u32).is_power_of_two()
            && (self.y() as u32).is_power_of_two()
    }

    #[inline]
    fn is_cube(&self) -> bool {
        self.x() == self.y()
    }

    #[inline]
    fn vector_div_floor(&self, rhs: &Self) -> Self {
        self.map_components_binary(rhs, |c1, c2| c1.div_floor(&c2))
    }

    #[inline]
    fn scalar_div_floor(&self, rhs: i32) -> Self {
        self.map_components_unary(|c| c.div_floor(&rhs))
    }

    #[inline]
    fn vector_div_ceil(&self, rhs: &Self) -> Self {
        self.map_components_binary(rhs, |c1, c2| c1.div_ceil(&c2))
    }

    #[inline]
    fn scalar_div_ceil(&self, rhs: i32) -> Self {
        self.map_components_unary(|c| c.div_ceil(&rhs))
    }
}

impl Neighborhoods for Point2i {
    #[inline]
    fn corner_offsets() -> Vec<Self> {
        vec![
            PointN([0, 0]),
            PointN([1, 0]),
            PointN([0, 1]),
            PointN([1, 1]),
        ]
    }

    #[inline]
    fn von_neumann_offsets() -> Vec<Self> {
        vec![
            PointN([-1, 0]),
            PointN([1, 0]),
            PointN([0, -1]),
            PointN([0, 1]),
        ]
    }

    #[inline]
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

/// An iterator over all points in an `Extent2<T>`.
pub struct Extent2PointIter<T>
where
    Range<T>: Iterator<Item = T>,
{
    product_iter: Product<Range<T>, Range<T>>,
}

impl<T> Iterator for Extent2PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    type Item = Point2<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.product_iter.next().map(|(y, x)| PointN([x, y]))
    }
}

impl IterExtent<[i32; 2]> for Point2i {
    type PointIter = Extent2PointIter<i32>;

    #[inline]
    fn iter_extent(min: &Point2i, lub: &Point2i) -> Self::PointIter {
        Extent2PointIter {
            // iproduct is opposite of row-major order.
            product_iter: iproduct!(min.y()..lub.y(), min.x()..lub.x()),
        }
    }
}

// This particular partial order allows us to say that an `Extent2i` e contains a `Point2i` p iff p
// is GEQ the minimum of e and p is LEQ the maximum of e.
impl<T> PartialOrd for Point2<T>
where
    T: Copy + PartialOrd,
{
    #[inline]
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

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.x() < other.x() && self.y() < other.y()
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.x() > other.x() && self.y() > other.y()
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.x() <= other.x() && self.y() <= other.y()
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.x() >= other.x() && self.y() >= other.y()
    }
}

// TODO: this could be defined on PointN once we have specialization
impl<T> Mul<T> for Point2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self {
        self.map_components_unary(|c| rhs * c)
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<i32> for Point2i {
    type Output = Self;

    #[inline]
    fn div(self, rhs: i32) -> Self {
        self.scalar_div_floor(rhs)
    }
}

impl Div<f32> for Point2f {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self {
        self.map_components_unary(|c| c / rhs)
    }
}

impl Div<Self> for Point2f {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.map_components_binary(&rhs, |c1, c2| c1 / c2)
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<Point2i> for Point2i {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Point2i) -> Self {
        self.vector_div_floor(&rhs)
    }
}

impl From<Point2i> for Point2f {
    #[inline]
    fn from(p: Point2i) -> Self {
        PointN([p.x() as f32, p.y() as f32])
    }
}

impl_componentwise_ops!(Point2i, i32);
impl_componentwise_ops!(Point2f, f32);
