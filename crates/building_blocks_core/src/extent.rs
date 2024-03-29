use crate::{point::point_traits::*, Point2, Point2f, Point3, Point3f, PointN};

use bytemuck::{Pod, Zeroable};
use core::ops::{Add, AddAssign, Mul, Shl, Shr, Sub, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2-dimensional extent with scalar type `T`.
pub type Extent2<T> = ExtentN<[T; 2]>;
/// A 2-dimensional extent with scalar type `i32`.
pub type Extent2i = ExtentN<[i32; 2]>;
/// A 2-dimensional extent with scalar type `f32`.
pub type Extent2f = ExtentN<[f32; 2]>;
/// A 3-dimensional extent with scalar type `T`.
pub type Extent3<T> = ExtentN<[T; 3]>;
/// A 3-dimensional extent with scalar type `i32`.
pub type Extent3i = ExtentN<[i32; 3]>;
/// A 3-dimensional extent with scalar type `f32`.
pub type Extent3f = ExtentN<[f32; 3]>;

/// An N-dimensional extent. This is mathematically the Cartesian product of a half-closed interval `[a, b)` in each dimension.
/// You can also just think of it as an axis-aligned box with some shape and a minimum point. When doing queries against lattice
/// maps, this is the primary structure used to determine the bounds of your query.
#[derive(Debug, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtentN<N> {
    /// The least point contained in the extent.
    pub minimum: PointN<N>,
    /// The length of each dimension.
    pub shape: PointN<N>,
}

unsafe impl<N> Zeroable for ExtentN<N> where PointN<N>: Zeroable {}
unsafe impl<N> Pod for ExtentN<N> where PointN<N>: Pod {}

// A few of these traits could be derived. But it seems that derive will not help the compiler infer trait bounds as well.

impl<N> Clone for ExtentN<N>
where
    PointN<N>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            minimum: self.minimum.clone(),
            shape: self.shape.clone(),
        }
    }
}
impl<N> Copy for ExtentN<N> where PointN<N>: Copy {}

impl<N> PartialEq for ExtentN<N>
where
    PointN<N>: PartialEq,
{
    // XXX: For some inconceivable reason, inlining this method hurts the performance of ChunkTree::get
    #[allow(clippy::missing_inline_in_public_items)]
    fn eq(&self, other: &Self) -> bool {
        self.minimum.eq(&other.minimum) && self.shape.eq(&other.shape)
    }
}

impl<N> ExtentN<N> {
    /// The default representation of an extent as the minimum point and shape.
    #[inline]
    pub fn from_min_and_shape(minimum: PointN<N>, shape: PointN<N>) -> Self {
        Self { minimum, shape }
    }
}

impl<N> ExtentN<N>
where
    PointN<N>: Point,
{
    #[inline]
    pub fn volume(&self) -> <PointN<N> as Point>::Scalar {
        self.shape.volume()
    }

    /// Translate the extent such that it has `new_min` as it's new minimum.
    #[inline]
    pub fn with_minimum(&self, new_min: PointN<N>) -> Self {
        Self::from_min_and_shape(new_min, self.shape)
    }

    /// The least point `p` for which all points `q` in the extent satisfy `q < p`.
    #[inline]
    pub fn least_upper_bound(&self) -> PointN<N> {
        self.minimum + self.shape
    }

    /// Returns `true` iff the point `p` is contained in this extent.
    #[inline]
    pub fn contains(&self, p: PointN<N>) -> bool {
        let lub = self.least_upper_bound();

        self.minimum <= p && p < lub
    }

    /// Resize the extent by mutating its `shape` by `delta`.
    #[inline]
    pub fn add_to_shape(&self, delta: PointN<N>) -> Self {
        Self::from_min_and_shape(self.minimum, self.shape + delta)
    }

    /// Returns a new extent that's been padded on all borders by `pad_amount`.
    #[inline]
    pub fn padded(&self, pad_amount: <PointN<N> as Point>::Scalar) -> Self
    where
        <PointN<N> as Point>::Scalar: Add<Output = <PointN<N> as Point>::Scalar>,
    {
        Self::from_min_and_shape(
            self.minimum - PointN::fill(pad_amount),
            self.shape + PointN::fill(pad_amount + pad_amount),
        )
    }

    /// Returns `Some(self)` iff this extent has a positive shape, otherise `None`.
    #[inline]
    pub fn check_positive_shape(self) -> Option<Self> {
        if self.shape <= PointN::ZERO {
            None
        } else {
            Some(self)
        }
    }
}

impl<N> ExtentN<N>
where
    PointN<N>: IntegerPoint,
{
    /// The number of points contained in the extent.
    #[inline]
    pub fn num_points(&self) -> usize {
        self.volume() as usize
    }

    /// Returns `true` iff the number of points in the extent is 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_points() == 0
    }

    /// An alternative representation of an extent as the minimum point and least upper bound.
    #[inline]
    pub fn from_min_and_lub(minimum: PointN<N>, least_upper_bound: PointN<N>) -> Self {
        let minimum = minimum;
        // We want to avoid negative shape components.
        let shape = (least_upper_bound - minimum).join(PointN::ZERO);

        Self { minimum, shape }
    }

    /// Returns the extent containing only the points in both `self` and `other`.
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        let minimum = self.minimum.join(other.minimum);
        let lub = self.least_upper_bound().meet(other.least_upper_bound());

        Self::from_min_and_lub(minimum, lub)
    }

    /// Returns the smallest extent containing all the points in `self` or `other`.
    ///
    /// This is not strictly a union of sets of points, but it is the closest we can get while still using an `ExtentN`.
    #[inline]
    pub fn quasi_union(&self, other: &Self) -> Self {
        let minimum = self.minimum.meet(other.minimum);
        let lub = self.least_upper_bound().join(other.least_upper_bound());

        Self::from_min_and_lub(minimum, lub)
    }

    /// Returns `true` iff the intersection of `self` and `other` is equal to `self`.
    #[inline]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.intersection(other).eq(self)
    }

    /// An alternative representation of an integer extent as the minimum point and maximum point. This only works for integer
    /// extents, where there is a unique maximum point.
    #[inline]
    pub fn from_min_and_max(minimum: PointN<N>, max: PointN<N>) -> Self {
        Self::from_min_and_lub(minimum, max + PointN::ONES)
    }

    /// The unique greatest point in the extent.
    #[inline]
    pub fn max(&self) -> PointN<N> {
        let lub = self.least_upper_bound();

        lub - PointN::ONES
    }

    /// Constructs the unique extent with both `p1` and `p2` as corners.
    #[inline]
    pub fn from_corners(p1: PointN<N>, p2: PointN<N>) -> Self {
        let min = p1.meet(p2);
        let max = p1.join(p2);

        Self::from_min_and_max(min, max)
    }

    /// Iterate over all points in the extent.
    /// ```
    /// # use building_blocks_core::prelude::*;
    /// #
    /// let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([2, 2, 1]));
    /// let points = extent.iter_points().collect::<Vec<_>>();
    /// assert_eq!(points, vec![
    ///     PointN([0, 0, 0]), PointN([1, 0, 0]), PointN([0, 1, 0]), PointN([1, 1, 0])
    /// ]);
    /// ```
    #[inline]
    pub fn iter_points(&self) -> <PointN<N> as IterExtent>::PointIter {
        PointN::iter_extent(self.minimum, self.least_upper_bound())
    }
}

impl<Nf, Ni> ExtentN<Nf>
where
    PointN<Nf>: FloatPoint<IntPoint = PointN<Ni>>,
    PointN<Ni>: IntegerPoint,
{
    /// Returns the integer `ExtentN` that contains `self`.
    #[inline]
    pub fn containing_integer_extent(&self) -> ExtentN<Ni> {
        ExtentN::from_min_and_max(
            self.minimum.floor_int(),
            self.least_upper_bound().floor_int(),
        )
    }
}

impl<N> Add<PointN<N>> for ExtentN<N>
where
    PointN<N>: Add<Output = PointN<N>>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: PointN<N>) -> Self::Output {
        ExtentN {
            minimum: self.minimum + rhs,
            shape: self.shape,
        }
    }
}

impl<N> Sub<PointN<N>> for ExtentN<N>
where
    PointN<N>: Sub<Output = PointN<N>>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: PointN<N>) -> Self::Output {
        ExtentN {
            minimum: self.minimum - rhs,
            shape: self.shape,
        }
    }
}

impl<N> Mul<PointN<N>> for ExtentN<N>
where
    PointN<N>: Copy + Mul<Output = PointN<N>>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: PointN<N>) -> Self::Output {
        ExtentN {
            minimum: self.minimum * rhs,
            shape: self.shape * rhs,
        }
    }
}

impl<N, T> Shl<T> for ExtentN<N>
where
    PointN<N>: Copy + Shl<T, Output = PointN<N>>,
    T: Copy,
{
    type Output = Self;

    #[inline]
    fn shl(self, rhs: T) -> Self::Output {
        ExtentN {
            minimum: self.minimum << rhs,
            shape: self.shape << rhs,
        }
    }
}

impl<N, T> Shr<T> for ExtentN<N>
where
    PointN<N>: Copy + Shr<T, Output = PointN<N>>,
    T: Copy,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: T) -> Self::Output {
        ExtentN {
            minimum: self.minimum >> rhs,
            shape: self.shape >> rhs,
        }
    }
}

impl<N> AddAssign<PointN<N>> for ExtentN<N>
where
    Self: Copy + Add<PointN<N>, Output = ExtentN<N>>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PointN<N>) {
        *self = *self + rhs;
    }
}

impl<N> SubAssign<PointN<N>> for ExtentN<N>
where
    Self: Copy + Sub<PointN<N>, Output = ExtentN<N>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PointN<N>) {
        *self = *self - rhs;
    }
}

/// Returns the smallest extent containing all of the given points.
#[inline]
pub fn bounding_extent<N, I>(mut points: I) -> ExtentN<N>
where
    I: Iterator<Item = PointN<N>>,
    PointN<N>: IntegerPoint,
{
    let first_point = points
        .next()
        .expect("Cannot find bounding extent of empty set of points");

    let mut min_point = first_point;
    let mut max_point = first_point;
    for p in points {
        min_point = min_point.meet(p);
        max_point = max_point.join(p);
    }

    ExtentN::from_min_and_max(min_point, max_point)
}

impl<T> Extent2<T>
where
    T: Copy,
    Point2<T>: Point,
{
    #[inline]
    pub fn corners(&self) -> [Point2<T>; 4] {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            PointN([min.x(), min.y()]),
            PointN([lub.x(), min.y()]),
            PointN([min.x(), lub.y()]),
            PointN([lub.x(), lub.y()]),
        ]
    }
}

impl<T> Extent3<T>
where
    T: Copy,
    Point3<T>: Point,
{
    #[inline]
    pub fn corners(&self) -> [Point3<T>; 8] {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            PointN([min.x(), min.y(), min.z()]),
            PointN([lub.x(), min.y(), min.z()]),
            PointN([min.x(), lub.y(), min.z()]),
            PointN([lub.x(), lub.y(), min.z()]),
            PointN([min.x(), min.y(), lub.z()]),
            PointN([lub.x(), min.y(), lub.z()]),
            PointN([min.x(), lub.y(), lub.z()]),
            PointN([lub.x(), lub.y(), lub.z()]),
        ]
    }
}

impl From<Extent2i> for Extent2f {
    #[inline]
    fn from(other: Extent2i) -> Self {
        Self::from_min_and_shape(Point2f::from(other.minimum), Point2f::from(other.shape))
    }
}

impl From<Extent3i> for Extent3f {
    #[inline]
    fn from(other: Extent3i) -> Self {
        Self::from_min_and_shape(Point3f::from(other.minimum), Point3f::from(other.shape))
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn row_major_extent_iter2() {
        let extent = Extent2i::from_min_and_shape(PointN([0, 0]), PointN([2, 2]));

        let points: Vec<_> = extent.iter_points().collect();

        assert_eq!(
            points,
            vec![
                PointN([0, 0]),
                PointN([1, 0]),
                PointN([0, 1]),
                PointN([1, 1]),
            ]
        );
    }

    #[test]
    fn row_major_extent_iter3() {
        let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([2, 2, 2]));

        let points: Vec<_> = extent.iter_points().collect();

        assert_eq!(
            points,
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
        );
    }

    #[test]
    fn empty_intersection_is_empty() {
        let e1 = Extent2i::from_min_and_max(PointN([0; 2]), PointN([1; 2]));
        let e2 = Extent2i::from_min_and_max(PointN([3; 2]), PointN([4; 2]));

        // A naive implementation might say the shape is [-1, -1].
        assert_eq!(e1.intersection(&e2).shape, PointN([0; 2]));
        assert!(e1.intersection(&e2).is_empty());
    }
}
