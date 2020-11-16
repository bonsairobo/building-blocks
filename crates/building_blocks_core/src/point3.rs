use crate::{
    point_traits::{GetComponent, Neighborhoods, SmallOne},
    Bounded, Distance, DotProduct, IntegerPoint, MapComponents, NormSquared, Ones, Point, Point2,
    PointN, SmallZero,
};

use core::ops::{Add, Div, Mul, Sub};
use num::{traits::Pow, Integer, Signed};
use std::cmp::Ordering;

/// A 3-dimensional point with scalar type `T`.
pub type Point3<T> = PointN<[T; 3]>;
/// A 3-dimensional point with scalar type `i32`.
pub type Point3i = PointN<[i32; 3]>;
/// A 3-dimensional point with scalar type `f32`.
pub type Point3f = PointN<[f32; 3]>;

impl<T> Point3<T> {
    #[inline]
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    #[inline]
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }

    #[inline]
    pub fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }

    #[inline]
    pub fn axis_component(&self, axis: Axis3) -> &T {
        &self.0[axis.index()]
    }
}

impl<T> Point3<T>
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
    pub fn z(&self) -> T {
        self.0[2]
    }

    #[inline]
    pub fn xy(&self) -> Point2<T> {
        PointN([self.x(), self.y()])
    }

    #[inline]
    pub fn yx(&self) -> Point2<T> {
        PointN([self.y(), self.x()])
    }

    #[inline]
    pub fn yz(&self) -> Point2<T> {
        PointN([self.y(), self.z()])
    }

    #[inline]
    pub fn zy(&self) -> Point2<T> {
        PointN([self.z(), self.y()])
    }

    #[inline]
    pub fn zx(&self) -> Point2<T> {
        PointN([self.z(), self.x()])
    }

    #[inline]
    pub fn xz(&self) -> Point2<T> {
        PointN([self.x(), self.z()])
    }

    #[inline]
    pub fn yzx(&self) -> Point3<T> {
        PointN([self.y(), self.z(), self.x()])
    }

    #[inline]
    pub fn zxy(&self) -> Point3<T> {
        PointN([self.z(), self.x(), self.y()])
    }

    #[inline]
    pub fn zyx(&self) -> Point3<T> {
        PointN([self.z(), self.y(), self.x()])
    }
}

impl<T> Point3<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T>,
{
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Self([
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        ])
    }
}

impl Point3f {
    #[inline]
    pub fn round(&self) -> Self {
        self.map_components_unary(|c| c.round())
    }

    #[inline]
    pub fn floor(&self) -> Self {
        self.map_components_unary(|c| c.floor())
    }
}

impl<T> Bounded for Point3<T>
where
    T: Bounded,
{
    const MAX: Self = PointN([T::MAX; 3]);
    const MIN: Self = PointN([T::MIN; 3]);
}

impl Point3i {
    #[inline]
    pub fn vector_div_floor(&self, rhs: &Self) -> Self {
        self.map_components_binary(rhs, |c1, c2| c1.div_floor(&c2))
    }

    #[inline]
    pub fn scalar_div_floor(&self, rhs: i32) -> Self {
        self.map_components_unary(|c| c.div_floor(&rhs))
    }
}

impl<T> MapComponents for Point3<T>
where
    T: Copy,
{
    type Scalar = T;

    #[inline]
    fn map_components_unary(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self {
        PointN([f(self.x()), f(self.y()), f(self.z())])
    }

    #[inline]
    fn map_components_binary(
        &self,
        other: &Self,
        f: impl Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self {
        PointN([
            f(self.x(), other.x()),
            f(self.y(), other.y()),
            f(self.z(), other.z()),
        ])
    }
}

impl<T> GetComponent for Point3<T>
where
    T: Copy,
{
    type Scalar = T;

    #[inline]
    fn at(&self, component_index: usize) -> T {
        self.0[component_index]
    }
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
    T: Copy + Signed + Add<Output = T> + Pow<u16, Output = T>,
    Point3<T>: Point<Scalar = T>,
{
    #[inline]
    fn l1_distance(&self, other: &Self) -> T {
        let diff = *self - *other;

        diff.x().abs() + diff.y().abs() + diff.z().abs()
    }

    #[inline]
    fn l2_distance_squared(&self, other: &Self) -> T {
        let diff = *self - *other;

        diff.x().pow(2) + diff.y().pow(2) + diff.z().pow(2)
    }
}

impl NormSquared for Point3i {
    #[inline]
    fn norm_squared(&self) -> f32 {
        self.dot(&self) as f32
    }
}

impl NormSquared for Point3f {
    #[inline]
    fn norm_squared(&self) -> f32 {
        self.dot(&self)
    }
}

impl<T> DotProduct for Point3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn dot(&self, other: &Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl IntegerPoint for Point3i {
    fn dimensions_are_powers_of_2(&self) -> bool {
        self.x().is_positive()
            && self.y().is_positive()
            && self.z().is_positive()
            && (self.x() as u32).is_power_of_two()
            && (self.y() as u32).is_power_of_two()
            && (self.z() as u32).is_power_of_two()
    }

    fn is_cube(&self) -> bool {
        self.x() == self.y() && self.x() == self.z()
    }
}

impl Neighborhoods for Point3i {
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

// This particular partial order allows us to say that an `Extent3i` e contains a `Point3i` p iff p
// is GEQ the minimum of e and p is LEQ the maximum of e.
impl<T> PartialOrd for Point3<T>
where
    T: Copy + PartialOrd,
{
    #[inline]
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

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.x() < other.x() && self.y() < other.y() && self.z() < other.z()
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.x() > other.x() && self.y() > other.y() && self.z() > other.z()
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.x() <= other.x() && self.y() <= other.y() && self.z() <= other.z()
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.x() >= other.x() && self.y() >= other.y() && self.z() >= other.z()
    }
}

impl<T> Mul<T> for Point3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self {
        self.map_components_unary(|c| rhs * c)
    }
}

impl<T> Mul<Point3<T>> for Point3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.map_components_binary(&other, |c1, c2| c1 * c2)
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<i32> for Point3i {
    type Output = Self;

    #[inline]
    fn div(self, rhs: i32) -> Self {
        self.scalar_div_floor(rhs)
    }
}

impl Div<f32> for Point3f {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self {
        self.map_components_unary(|c| c / rhs)
    }
}

impl Div<Self> for Point3f {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.map_components_binary(&rhs, |c1, c2| c1 / c2)
    }
}

// Use specialized implementation for integers because the default Div impl rounds towards zero,
// which is not what we want.
impl Div<Self> for Point3i {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Point3i) -> Self {
        self.vector_div_floor(&rhs)
    }
}

impl From<Point3i> for Point3f {
    #[inline]
    fn from(p: Point3i) -> Self {
        PointN([p.x() as f32, p.y() as f32, p.z() as f32])
    }
}

impl Point3f {
    #[inline]
    pub fn as_3i(&self) -> Point3i {
        PointN([self.x() as i32, self.y() as i32, self.z() as i32])
    }

    #[inline]
    pub fn in_voxel(&self) -> Point3i {
        self.floor().as_3i()
    }
}

#[cfg(feature = "mint")]
pub mod mint_conversions {
    use super::*;

    impl<T> From<mint::Point3<T>> for Point3<T> {
        #[inline]
        fn from(p: mint::Point3<T>) -> Self {
            PointN([p.x, p.y, p.z])
        }
    }

    impl<T> From<Point3<T>> for mint::Point3<T>
    where
        T: Clone,
    {
        #[inline]
        fn from(p: Point3<T>) -> Self {
            mint::Point3::from_slice(&p.0)
        }
    }

    impl<T> From<mint::Vector3<T>> for Point3<T> {
        #[inline]
        fn from(p: mint::Vector3<T>) -> Self {
            PointN([p.x, p.y, p.z])
        }
    }

    impl<T> From<Point3<T>> for mint::Vector3<T>
    where
        T: Clone,
    {
        #[inline]
        fn from(p: Point3<T>) -> Self {
            mint::Vector3::from_slice(&p.0)
        }
    }
}

/// Either the X, Y, or Z axis.
#[derive(Clone, Copy)]
pub enum Axis3 {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis3 {
    /// The index for a point's component on this axis.
    pub fn index(&self) -> usize {
        *self as usize
    }
}
