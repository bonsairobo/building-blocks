use crate::Axis3;

use super::{point_traits::*, Point2, PointN};

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Range, Rem, Shl, Shr, Sub};
use itertools::{iproduct, ConsTuples, Product};
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

    #[inline]
    pub fn axis_component_mut(&mut self, axis: Axis3) -> &mut T {
        &mut self.0[axis.index()]
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
    pub fn yzx(&self) -> Self {
        PointN([self.y(), self.z(), self.x()])
    }

    #[inline]
    pub fn zxy(&self) -> Self {
        PointN([self.z(), self.x(), self.y()])
    }

    #[inline]
    pub fn zyx(&self) -> Self {
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

impl Point3i {
    pub const CUBE_CORNER_OFFSETS: [Self; 8] = [
        PointN([0, 0, 0]),
        PointN([1, 0, 0]),
        PointN([0, 1, 0]),
        PointN([1, 1, 0]),
        PointN([0, 0, 1]),
        PointN([1, 0, 1]),
        PointN([0, 1, 1]),
        PointN([1, 1, 1]),
    ];

    pub const VON_NEUMANN_OFFSETS: [Self; 6] = [
        PointN([-1, 0, 0]),
        PointN([1, 0, 0]),
        PointN([0, -1, 0]),
        PointN([0, 1, 0]),
        PointN([0, 0, -1]),
        PointN([0, 0, 1]),
    ];

    // Because there are "moore" of them... huehuehue
    pub const MOORE_OFFSETS: [Self; 26] = [
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
    ];
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

impl<T> Bounded for Point3<T>
where
    T: Bounded,
{
    const MAX: Self = PointN([T::MAX; 3]);
    const MIN: Self = PointN([T::MIN; 3]);
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

    #[inline]
    fn fill(value: i32) -> Self {
        Self([value; 3])
    }

    #[inline]
    fn basis() -> Vec<Self> {
        vec![PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([0, 0, 1])]
    }

    #[inline]
    fn volume(&self) -> <Self as Point>::Scalar {
        self.x() * self.y() * self.z()
    }
}

impl Point for Point3f {
    type Scalar = f32;

    #[inline]
    fn fill(value: f32) -> Self {
        Self([value; 3])
    }

    #[inline]
    fn basis() -> Vec<Self> {
        vec![
            PointN([1.0, 0.0, 0.0]),
            PointN([0.0, 1.0, 0.0]),
            PointN([0.0, 0.0, 1.0]),
        ]
    }

    #[inline]
    fn volume(&self) -> <Self as Point>::Scalar {
        self.x() * self.y()
    }
}

impl<T> ConstZero for Point3<T>
where
    T: ConstZero,
{
    const ZERO: Self = PointN([T::ZERO; 3]);
}

impl<T> Ones for Point3<T>
where
    T: ConstOne,
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

impl IntegerPoint<[i32; 3]> for Point3i {
    #[inline]
    fn dimensions_are_powers_of_2(&self) -> bool {
        self.x().is_positive()
            && self.y().is_positive()
            && self.z().is_positive()
            && (self.x() as u32).is_power_of_two()
            && (self.y() as u32).is_power_of_two()
            && (self.z() as u32).is_power_of_two()
    }

    #[inline]
    fn is_cube(&self) -> bool {
        self.x() == self.y() && self.x() == self.z()
    }
}

impl Neighborhoods for Point3i {
    #[inline]
    fn corner_offsets() -> Vec<Self> {
        Self::CUBE_CORNER_OFFSETS.iter().cloned().collect()
    }

    #[inline]
    fn von_neumann_offsets() -> Vec<Self> {
        Self::VON_NEUMANN_OFFSETS.iter().cloned().collect()
    }

    #[inline]
    fn moore_offsets() -> Vec<Self> {
        Self::MOORE_OFFSETS.iter().cloned().collect()
    }
}

/// An iterator over all points in an `Extent3<T>`.
pub struct Extent3PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    product_iter: ConsTuples<RangeProduct3<T>, ((T, T), T)>,
}

type RangeProduct2<T> = Product<Range<T>, Range<T>>;
type RangeProduct3<T> = Product<RangeProduct2<T>, Range<T>>;

impl<T> Iterator for Extent3PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    type Item = Point3<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.product_iter.next().map(|(z, y, x)| PointN([x, y, z]))
    }
}

impl IterExtent<[i32; 3]> for Point3i {
    type PointIter = Extent3PointIter<i32>;

    #[inline(always)]
    fn iter_extent(min: &Point3i, lub: &Point3i) -> Self::PointIter {
        Extent3PointIter {
            // iproduct is opposite of row-major order.
            product_iter: iproduct!(min.z()..lub.z(), min.y()..lub.y(), min.x()..lub.x()),
        }
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

impl From<Point3i> for Point3f {
    #[inline]
    fn from(p: Point3i) -> Self {
        PointN([p.x() as f32, p.y() as f32, p.z() as f32])
    }
}

impl_unary_ops!(Point3f, f32);
impl_unary_ops!(Point3i, i32);

impl_binary_ops!(Point3i, i32);
impl_binary_ops!(Point3f, f32);

impl_unary_float_ops!(Point3f);
impl_unary_integer_ops!(Point3i, i32);

impl_binary_integer_ops!(Point3i);

impl_float_div!(Point3f, f32);
impl_integer_div!(Point3i, i32);
