use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use num::Zero;
use serde::{Deserialize, Serialize};

/// An N-dimensional point (where N=2 or N=3), which is usually just a primitive array like
/// `[i32; 2]` or `[i32; 3]`. It is most convenient to construct points of any dimension as:
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
/// # use building_blocks_core::PointN;
/// #
/// let p1 = PointN([1, 2]);
/// let p2 = PointN([3, 4]);
///
/// assert_eq!(p1 + p2, PointN([4, 6]));
/// assert_eq!(p1 - p2, PointN([-2, -2]));
///
/// assert_eq!(p1 * 2, PointN([2, 4]));
/// assert_eq!(p1 / 2, PointN([0, 1]));
///
/// // Also some component-wise operations.
/// assert_eq!(p1 * p2, PointN([3, 8]));
/// assert_eq!(p1 / p2, PointN([0, 0]));
/// assert_eq!(p2 / p1, PointN([3, 2]));
/// ```
///
/// There is also a partial order defined on points which says that a point A is greater than a
/// point B if and only if all of the components of point A are greater than point B. This is useful
/// for easily checking is a point is inside of the extent between two other points:
///
/// ```
/// # use building_blocks_core::PointN;
/// #
/// let min = PointN([0, 0, 0]);
/// let least_upper_bound = PointN([3, 3, 3]);
///
/// let p = PointN([0, 1, 2]);
/// assert!(min <= p && p < least_upper_bound);
/// ```
#[derive(Copy, Clone, Debug, Deserialize, Default, Eq, Hash, PartialEq, Serialize)]
pub struct PointN<N>(pub N);

pub trait MapComponents {
    type Scalar: Copy;

    /// Returns the point after applying `f` component-wise.
    fn map_components_unary(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self;

    /// Returns the point after applying `f` component-wise to both `self` and `other` in parallel.
    fn map_components_binary(
        &self,
        other: &Self,
        f: impl Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self;
}

/// A trait that bundles op traits that all `PointN<N>` (and its components) should have.
pub trait Point:
    Add<Output = Self>
    + Bounded
    + Copy
    + Div<<Self as Point>::Scalar, Output = Self>
    + Div<Self, Output = Self>
    + MapComponents<Scalar = <Self as Point>::Scalar>
    + Mul<<Self as Point>::Scalar, Output = Self>
    + Mul<Self, Output = Self>
    + Ones
    + PartialOrd
    + Sized
    + Sub<Output = Self>
    + Neg
    + Zero
{
    type Scalar: Copy;

    fn basis() -> Vec<Self>;

    /// Returns a point where each component is the absolute value of the input component.
    fn abs(&self) -> Self;

    /// Returns the component specified by index. I.e. X = 0, Y = 1, Z = 2.
    fn at(&self, component_index: usize) -> <Self as Point>::Scalar;
}

impl<N> Neg for PointN<N>
where
    N: Copy,
    PointN<N>: Sub<Output = Self> + Zero,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl<N> AddAssign for PointN<N>
where
    N: Copy,
    PointN<N>: Add<Output = Self>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<N> SubAssign for PointN<N>
where
    N: Copy,
    PointN<N>: Sub<Output = Self>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

pub trait Ones: Copy {
    /// A point of all ones.
    const ONES: Self;
}

pub trait Distance: Point {
    /// The L1 distance between points.
    fn l1_distance(&self, other: &Self) -> <Self as Point>::Scalar;

    /// The square of the L2 (Euclidean) distance between points.
    fn l2_distance_squared(&self, other: &Self) -> <Self as Point>::Scalar;
}

pub trait NormSquared {
    fn norm_squared(&self) -> f32;
}

pub trait Norm {
    fn norm(&self) -> f32;
}

impl<T> Norm for T
where
    T: NormSquared,
{
    #[inline]
    fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }
}

impl<N> Zero for PointN<N>
where
    Self: Point + SmallZero,
{
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

pub trait DotProduct {
    type Scalar: Copy;

    /// The vector dot product.
    fn dot(&self, other: &Self) -> Self::Scalar;
}

pub trait IntegerPoint: Bounded + Point {
    /// Component-wise maximum.
    fn join(&self, other: &Self) -> Self;

    /// Component-wise minimum.
    fn meet(&self, other: &Self) -> Self;

    /// Left bitshifts all dimensions.
    fn scalar_left_shift(&self, shift_by: <Self as Point>::Scalar) -> Self;

    /// Right bitshifts all dimensions.
    fn scalar_right_shift(&self, shift_by: <Self as Point>::Scalar) -> Self;

    /// Left bitshifts all dimensions, component-wise.
    fn vector_left_shift(&self, shift_by: &Self) -> Self;

    /// Right bitshifts all dimensions, component-wise.
    fn vector_right_shift(&self, shift_by: &Self) -> Self;

    /// All corners of an N-dimensional unit cube.
    fn corner_offsets() -> Vec<Self>;

    /// [Von Neumann Neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
    fn von_neumann_offsets() -> Vec<Self>;

    /// [Moore Neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
    fn moore_offsets() -> Vec<Self>;

    /// Returns `true` iff all dimensions are powers of 2.
    fn dimensions_are_powers_of_2(&self) -> bool;

    /// Returns `true` iff all dimensions are equal.
    fn is_cube(&self) -> bool;
}

macro_rules! componentwise_integer_point_impl {
    () => {
        #[inline]
        fn join(&self, other: &Self) -> Self {
            self.map_components_binary(other, |c1, c2| max(c1, c2))
        }

        #[inline]
        fn meet(&self, other: &Self) -> Self {
            self.map_components_binary(other, |c1, c2| min(c1, c2))
        }

        #[inline]
        fn scalar_left_shift(&self, shift_by: i32) -> Self {
            self.map_components_unary(|c| c << shift_by)
        }

        #[inline]
        fn scalar_right_shift(&self, shift_by: i32) -> Self {
            self.map_components_unary(|c| c >> shift_by)
        }

        #[inline]
        fn vector_left_shift(&self, shift_by: &Self) -> Self {
            self.map_components_binary(shift_by, |c1, c2| c1 << c2)
        }

        #[inline]
        fn vector_right_shift(&self, shift_by: &Self) -> Self {
            self.map_components_binary(shift_by, |c1, c2| c1 >> c2)
        }
    };
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

impl SmallZero for f32 {
    const ZERO: f32 = 0.0;
}
impl SmallOne for f32 {
    const ONE: f32 = 1.0;
}

pub trait Bounded: Copy {
    const MIN: Self;
    const MAX: Self;
}

impl Bounded for i32 {
    const MIN: Self = std::i32::MIN;
    const MAX: Self = std::i32::MAX;
}

impl Bounded for f32 {
    const MIN: Self = std::f32::MIN;
    const MAX: Self = std::f32::MAX;
}
