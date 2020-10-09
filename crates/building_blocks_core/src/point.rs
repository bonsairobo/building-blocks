use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use num::Zero;
use serde::{Deserialize, Serialize};

/// An N-dimensional point (where N=2 or N=3), which is usually just a primitive array of type `D`.
/// It is most convenient to construct points of any dimension as:
///
/// ```
/// use building_blocks_core::PointN;
///
/// let p2 = PointN([1, 2]); // 2D
/// let p3 = PointN([1, 2, 3]); // 3D
/// ```
///
/// Points support basic linear algebraic operations such as addition, subtraction, scalar
/// multiplication, and scalar division.
///
/// ```
/// use building_blocks_core::PointN;
///
/// let p1 = PointN([1, 2]);
/// let p2 = PointN([3, 4]);
///
/// assert_eq!(p1 + p2, PointN([4, 6]));
/// assert_eq!(p1 - p2, PointN([-2, -2]));
///
/// assert_eq!(p1 * 2, PointN([2, 4]));
/// assert_eq!(p1 / 2, PointN([0, 1]));
/// ```
///
/// There is also a partial order defined on points which says that a point A is greater than a
/// point B if and only if all of the components of point A are greater than point B. This is useful
/// for easily checking is a point is inside of the extent between two other points:
///
/// ```
/// use building_blocks_core::PointN;
///
/// let min = PointN([0, 0, 0]);
/// let least_upper_bound = PointN([3, 3, 3]);
///
/// let p = PointN([0, 1, 2]);
/// assert!(min <= p && p < least_upper_bound);
/// ```
#[derive(Copy, Clone, Debug, Deserialize, Default, Eq, Hash, PartialEq, Serialize)]
pub struct PointN<N>(pub N);

/// A subtrait that bundles op traits that all `PointN<N>` (and its components) should have.
pub trait PointOps:
    Copy
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<<Self as PointOps>::Scalar, Output = Self>
    + Div<<Self as PointOps>::Scalar, Output = Self>
    + PartialOrd
    + Sized
{
    type Scalar: Copy;
}

/// A subtrait of `PointOps` which also includes methods that must be implemented specially by each
/// `PointN<N>`. The goal is only to generalize over the number of dimensions.
pub trait Point: PointOps {
    /// A point of all ones.
    const ONES: Self;
    /// A point of all zeros, algebraically the "zero element."
    const ZERO: Self;

    /// The least point.
    const MIN: Self;
    /// The greatest point.
    const MAX: Self;

    /// Component-wise maximum.
    fn join(&self, other: &Self) -> Self;

    /// Component-wise minimum.
    fn meet(&self, other: &Self) -> Self;

    /// The vector dot product.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// The L1 distance between points.
    fn l1_distance(&self, other: &Self) -> Self::Scalar;

    /// The square of the L2 (Euclidean) distance between points.
    fn l2_distance_squared(&self, other: &Self) -> Self::Scalar;
}

impl<N> Zero for PointN<N>
where
    PointN<N>: Point,
{
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

pub trait IntegerPoint: Point {
    fn corner_offsets() -> Vec<Self>;

    /// https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
    fn von_neumann_offsets() -> Vec<Self>;

    /// https://en.wikipedia.org/wiki/Moore_neighborhood
    fn moore_offsets() -> Vec<Self>;
}

impl<N> AddAssign for PointN<N>
where
    N: Copy,
    PointN<N>: Add<Output = Self>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<N> SubAssign for PointN<N>
where
    N: Copy,
    PointN<N>: Sub<Output = Self>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

// `Zero` trait doesn't allow associated constants for zero because of bignums.
pub trait SmallZero: Copy {
    const ZERO: Self;
}

// `One` trait doesn't allow associated constants for one because of bignums.
pub trait SmallOne: Copy {
    const ONE: Self;
}

impl SmallZero for i32 {
    const ZERO: i32 = 0;
}
impl SmallOne for i32 {
    const ONE: i32 = 1;
}

pub trait Bounded: Copy {
    const MIN: Self;
    const MAX: Self;
}

impl Bounded for i32 {
    const MIN: Self = std::i32::MIN;
    const MAX: Self = std::i32::MAX;
}
