use crate::{
    point_traits::{Bounded, ComponentwiseIntegerOps, IntegerPoint, Ones, Point},
    PointN,
};

use core::ops::{Add, AddAssign, Sub, SubAssign};
use serde::{Deserialize, Serialize};

/// An N-dimensional extent. This is mathematically the Cartesian product of a half-closed interval
/// `[a, b)` in each dimension. You can also just think of it as an axis-aligned box with some shape
/// and a minimum point. When doing queries against lattice maps, this is the primary structure used
/// to determine the bounds of your query.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExtentN<N> {
    /// The least point contained in the extent.
    pub minimum: PointN<N>,
    /// The length of each dimension.
    pub shape: PointN<N>,
}

impl<N> ExtentN<N> {
    /// The default representation of an extent as the minimum point and shape.
    pub fn from_min_and_shape(minimum: PointN<N>, shape: PointN<N>) -> Self {
        Self { minimum, shape }
    }
}

impl<N> ExtentN<N>
where
    PointN<N>: Point,
{
    /// An alternative representation of an extent as the minimum point and least upper bound.
    pub fn from_min_and_lub(minimum: PointN<N>, least_upper_bound: PointN<N>) -> Self {
        let minimum = minimum;
        let shape = least_upper_bound - minimum;

        Self { minimum, shape }
    }

    /// Translate the extent such that it has `new_min` as it's new minimum.
    pub fn with_minimum(&self, new_min: PointN<N>) -> Self {
        Self::from_min_and_shape(new_min, self.shape)
    }

    /// The least point `p` for which all points `q` in the extent satisfy `q < p`.
    pub fn least_upper_bound(&self) -> PointN<N> {
        self.minimum + self.shape
    }

    /// Returns `true` iff the point `p` is contained in this extent.
    pub fn contains(&self, p: &PointN<N>) -> bool {
        let lub = self.least_upper_bound();

        self.minimum <= *p && *p < lub
    }

    pub fn add_to_shape(&self, delta: PointN<N>) -> Self {
        Self::from_min_and_shape(self.minimum, self.shape + delta)
    }

    pub fn padded(&self, pad_amount: <PointN<N> as Point>::Scalar) -> Self
    where
        PointN<N>: Ones,
        <PointN<N> as Point>::Scalar: Add<Output = <PointN<N> as Point>::Scalar>,
    {
        Self::from_min_and_shape(
            self.minimum - (PointN::ONES * pad_amount),
            self.shape + (PointN::ONES * (pad_amount + pad_amount)),
        )
    }
}

impl<N> ExtentN<N>
where
    PointN<N>: IntegerPoint,
{
    /// Returns the extent containing only the points in both `self` and `other`.
    pub fn intersection(&self, other: &Self) -> Self {
        let minimum = self.minimum.join(&other.minimum);
        let lub = self.least_upper_bound().meet(&other.least_upper_bound());

        Self::from_min_and_lub(minimum, lub)
    }

    pub fn is_subset_of(&self, other: &Self) -> bool
    where
        // TODO: seems like the compiler ought to infer this is true when PointN<N>: PartialEq
        Self: PartialEq,
    {
        self.intersection(other).eq(self)
    }
}

impl<N> ExtentN<N>
where
    PointN<N>: Point + Ones,
    ExtentN<N>: IntegerExtent<N>,
{
    /// An alternative representation of an integer extent as the minimum point and maximum point.
    /// This only works for integer extents, where there is a unique maximum point.
    pub fn from_min_and_max(minimum: PointN<N>, max: PointN<N>) -> Self {
        Self::from_min_and_lub(minimum, max + PointN::ONES)
    }

    /// The unique greatest point in the extent.
    pub fn max(&self) -> PointN<N> {
        let lub = self.least_upper_bound();

        lub - PointN::ONES
    }
}

/// A trait for methods that all `ExtentN<N>` should have, but only those which are implemented
/// specially for each `ExtentN<N>`. The goal is only to generalize over the number of dimensions.
pub trait Extent<N> {
    type VolumeType;

    /// For integer extents, the number of points in the extent.
    fn volume(&self) -> Self::VolumeType;
}

/// The methods that all `ExtentN<N>` should have when the scalar type is an integer, but only those
/// which are implemented specially for each `ExtentN<N>`. This enables us to assume that any finite
/// extent contains only a finite number of points.
pub trait IntegerExtent<N>: Extent<N> + Copy {
    type PointIter: Iterator<Item = PointN<N>>;

    /// The number of points contained in the extent.
    fn num_points(&self) -> usize;

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
    fn iter_points(&self) -> Self::PointIter;
}

impl<T> Add<PointN<T>> for ExtentN<T>
where
    PointN<T>: Add<Output = PointN<T>>,
{
    type Output = Self;

    fn add(self, rhs: PointN<T>) -> Self::Output {
        ExtentN {
            minimum: self.minimum + rhs,
            shape: self.shape,
        }
    }
}

impl<T> Sub<PointN<T>> for ExtentN<T>
where
    PointN<T>: Sub<Output = PointN<T>>,
{
    type Output = Self;

    fn sub(self, rhs: PointN<T>) -> Self::Output {
        ExtentN {
            minimum: self.minimum - rhs,
            shape: self.shape,
        }
    }
}

impl<T> AddAssign<PointN<T>> for ExtentN<T>
where
    Self: Copy + Add<PointN<T>, Output = ExtentN<T>>,
{
    fn add_assign(&mut self, rhs: PointN<T>) {
        *self = *self + rhs;
    }
}

impl<T> SubAssign<PointN<T>> for ExtentN<T>
where
    Self: Copy + Sub<PointN<T>, Output = ExtentN<T>>,
{
    fn sub_assign(&mut self, rhs: PointN<T>) {
        *self = *self - rhs;
    }
}

/// Returns the smallest extent containing all of the given points.
pub fn bounding_extent<N, I>(points: I) -> ExtentN<N>
where
    I: Iterator<Item = PointN<N>>,
    PointN<N>: IntegerPoint,
    ExtentN<N>: IntegerExtent<N>,
{
    let mut min_point = PointN::MAX;
    let mut max_point = PointN::MIN;
    for p in points {
        min_point = min_point.meet(&p);
        max_point = max_point.join(&p);
    }

    ExtentN::from_min_and_max(min_point, max_point)
}
