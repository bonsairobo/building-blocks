use crate::{Axis2, Morton2};

use super::{point_traits::*, PointN};

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Range, Rem, Shl, Shr};
use itertools::{iproduct, Product};
use num::{traits::Pow, Integer, Signed};
use std::cmp::Ordering;

/// A 2-dimensional point with scalar type `T`.
pub type Point2<T> = PointN<[T; 2]>;
/// A 2-dimensional point with scalar type `i32`.
pub type Point2i = Point2<i32>;
/// A 2-dimensional point with scalar type `f32`.
pub type Point2f = Point2<f32>;

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
    pub fn axis_component_mut(&mut self, axis: Axis2) -> &mut T {
        &mut self.0[axis.index()]
    }
}

impl<T> Point2<T>
where
    T: Copy,
{
    #[inline]
    pub fn axis_component(self, axis: Axis2) -> T {
        self.0[axis.index()]
    }

    #[inline]
    pub fn x(self) -> T {
        self.0[0]
    }

    #[inline]
    pub fn y(self) -> T {
        self.0[1]
    }

    #[inline]
    pub fn yx(self) -> Self {
        PointN([self.y(), self.x()])
    }
}

impl Point2i {
    pub const SQUARE_CORNER_OFFSETS: [Self; 4] = [
        PointN([0, 0]),
        PointN([1, 0]),
        PointN([0, 1]),
        PointN([1, 1]),
    ];

    pub const VON_NEUMANN_OFFSETS: [Self; 4] = [
        PointN([-1, 0]),
        PointN([1, 0]),
        PointN([0, -1]),
        PointN([0, 1]),
    ];

    pub const MOORE_OFFSETS: [Self; 8] = [
        PointN([-1, -1]),
        PointN([0, -1]),
        PointN([1, -1]),
        PointN([-1, 0]),
        PointN([1, 0]),
        PointN([-1, 1]),
        PointN([0, 1]),
        PointN([1, 1]),
    ];
}

impl Point2f {
    /// Returns the coordinates of the pixel containing `self`.
    #[inline]
    pub fn in_pixel(self) -> Point2i {
        self.floor_int()
    }
}

impl IntoIntegerPoint for Point2f {
    type IntPoint = Point2i;

    #[inline]
    fn into_int(self) -> Self::IntPoint {
        PointN([self.x() as i32, self.y() as i32])
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
    fn map_components_unary(self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self {
        PointN([f(self.x()), f(self.y())])
    }

    #[inline]
    fn map_components_binary(
        self,
        other: Self,
        f: impl Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self {
        PointN([f(self.x(), other.x()), f(self.y(), other.y())])
    }
}

impl MinMaxComponent for Point2i {
    type Scalar = i32;

    #[inline]
    fn min_component(self) -> Self::Scalar {
        self.x().min(self.y())
    }
    #[inline]
    fn max_component(self) -> Self::Scalar {
        self.x().max(self.y())
    }
}

impl MinMaxComponent for Point2f {
    type Scalar = f32;

    #[inline]
    fn min_component(self) -> Self::Scalar {
        self.x().min(self.y())
    }
    #[inline]
    fn max_component(self) -> Self::Scalar {
        self.x().max(self.y())
    }
}

impl<T> GetComponent for Point2<T>
where
    T: Copy,
{
    type Scalar = T;

    #[inline]
    fn at(self, component_index: usize) -> T {
        self.0[component_index]
    }
}

impl Point for Point2i {
    type Scalar = i32;

    #[inline]
    fn fill(value: i32) -> Self {
        Self([value; 2])
    }

    #[inline]
    fn volume(self) -> <Self as Point>::Scalar {
        self.x() * self.y()
    }
}

impl Point for Point2f {
    type Scalar = f32;

    #[inline]
    fn fill(value: f32) -> Self {
        Self([value; 2])
    }

    #[inline]
    fn volume(self) -> <Self as Point>::Scalar {
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
    fn l1_distance(self, other: Self) -> T {
        let diff = self - other;

        diff.x().abs() + diff.y().abs()
    }

    #[inline]
    fn l2_distance_squared(self, other: Self) -> T {
        let diff = self - other;

        diff.x().pow(2) + diff.y().pow(2)
    }
}

impl Norm for Point2i {
    #[inline]
    fn norm_squared(self) -> f32 {
        self.dot(self) as f32
    }
}

impl Norm for Point2f {
    #[inline]
    fn norm_squared(self) -> f32 {
        self.dot(self)
    }
}

impl<T> DotProduct for Point2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn dot(self, other: Self) -> Self::Scalar {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl IntegerPoint for Point2i {
    type Morton = Morton2;
    type FloatPoint = Point2f;

    #[inline]
    fn dimensions_are_powers_of_2(self) -> bool {
        self.x().is_positive()
            && self.y().is_positive()
            && (self.x() as u32).is_power_of_two()
            && (self.y() as u32).is_power_of_two()
    }

    #[inline]
    fn is_cube(self) -> bool {
        self.x() == self.y()
    }
}

impl Neighborhoods for Point2i {
    const NUM_CORNERS: u8 = 4;

    #[inline]
    fn corner_offsets() -> Vec<Self> {
        Self::SQUARE_CORNER_OFFSETS.to_vec()
    }

    #[inline]
    fn corner_offset(index: u8) -> Self {
        debug_assert!(index < Self::NUM_CORNERS);
        Self::SQUARE_CORNER_OFFSETS[index as usize]
    }

    #[inline]
    fn as_corner_index(&self) -> u8 {
        self.x() as u8 | ((self.y() as u8) << 1)
    }

    #[inline]
    fn von_neumann_offsets() -> Vec<Self> {
        Self::VON_NEUMANN_OFFSETS.to_vec()
    }

    #[inline]
    fn moore_offsets() -> Vec<Self> {
        Self::MOORE_OFFSETS.to_vec()
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

impl IterExtent for Point2i {
    type PointIter = Extent2PointIter<i32>;

    #[inline]
    fn iter_extent(min: Point2i, lub: Point2i) -> Self::PointIter {
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

impl From<Point2i> for Point2f {
    #[inline]
    fn from(p: Point2i) -> Self {
        PointN([p.x() as f32, p.y() as f32])
    }
}

impl_unary_ops!(Point2f, f32);
impl_unary_ops!(Point2i, i32);

impl_binary_ops!(Point2i, i32);
impl_binary_ops!(Point2f, f32);

impl_unary_integer_ops!(Point2i, i32);

impl_shr_shl!(Point2i, i8);
impl_shr_shl!(Point2i, i16);
impl_shr_shl!(Point2i, i32);
impl_shr_shl!(Point2i, u8);
impl_shr_shl!(Point2i, u16);
impl_shr_shl!(Point2i, u32);

impl_binary_integer_ops!(Point2i);

impl_float_div!(Point2f, f32);
impl_integer_div!(Point2i, i32);

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_index_matches_offsets() {
        for c in 0..Point2i::NUM_CORNERS {
            assert_eq!(Point2i::corner_offset(c).as_corner_index(), c);
        }
    }
}
