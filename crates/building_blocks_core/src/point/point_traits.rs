use crate::PointN;

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use num::Zero;

/// A trait that bundles op traits that all `PointN<N>` (and its components) should have.
pub trait Point:
    'static
    + Abs
    + Add<Output = Self>
    + AddAssign
    + Bounded
    + Copy
    + Div<<Self as Point>::Scalar, Output = Self>
    + Div<Self, Output = Self>
    + GetComponent<Scalar = <Self as Point>::Scalar>
    + MapComponents<Scalar = <Self as Point>::Scalar>
    + Mul<<Self as Point>::Scalar, Output = Self>
    + Mul<Self, Output = Self>
    + Ones
    + PartialEq
    + PartialOrd
    + Sized
    + Sub<Output = Self>
    + SubAssign
    + Neg
    + Zero
{
    type Scalar: Copy;

    fn basis() -> Vec<Self>;

    fn volume(&self) -> <Self as Point>::Scalar;
}

pub trait Abs {
    fn abs(&self) -> Self;
}

pub trait GetComponent {
    type Scalar: Copy;

    /// Returns the component specified by index. I.e. X = 0, Y = 1, Z = 2.
    fn at(&self, component_index: usize) -> Self::Scalar;
}

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

pub trait DotProduct {
    type Scalar: Copy;

    /// The vector dot product.
    fn dot(&self, other: &Self) -> Self::Scalar;
}

pub trait IntegerPoint<N>:
    ComponentwiseIntegerOps + Eq + IterExtent<N> + Neighborhoods + Point<Scalar = i32>
{
    /// Returns `true` iff all dimensions are powers of 2.
    fn dimensions_are_powers_of_2(&self) -> bool;

    /// Returns `true` iff all dimensions are equal.
    fn is_cube(&self) -> bool;
}

pub trait ComponentwiseOps {
    /// Component-wise maximum.
    fn join(&self, other: &Self) -> Self;

    /// Component-wise minimum.
    fn meet(&self, other: &Self) -> Self;
}

macro_rules! impl_componentwise_ops {
    ($t:ty, $scalar:ty) => {
        impl ComponentwiseOps for $t {
            #[inline]
            fn join(&self, other: &Self) -> Self {
                self.map_components_binary(other, <$scalar>::max)
            }

            #[inline]
            fn meet(&self, other: &Self) -> Self {
                self.map_components_binary(other, <$scalar>::min)
            }
        }
    };
}

pub trait ComponentwiseIntegerOps: ComponentwiseOps + Point {
    /// Left bitshifts all dimensions.
    fn scalar_left_shift(&self, shift_by: <Self as Point>::Scalar) -> Self;

    /// Right bitshifts all dimensions.
    fn scalar_right_shift(&self, shift_by: <Self as Point>::Scalar) -> Self;

    /// Left bitshifts all dimensions, component-wise.
    fn vector_left_shift(&self, shift_by: &Self) -> Self;

    /// Right bitshifts all dimensions, component-wise.
    fn vector_right_shift(&self, shift_by: &Self) -> Self;
}

impl<T> ComponentwiseIntegerOps for T
where
    T: MapComponents<Scalar = i32> + Point<Scalar = i32> + ComponentwiseOps,
{
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
}

pub trait Neighborhoods: Sized {
    /// All corners of an N-dimensional unit cube.
    fn corner_offsets() -> Vec<Self>;

    /// [Von Neumann Neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
    fn von_neumann_offsets() -> Vec<Self>;

    /// [Moore Neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
    fn moore_offsets() -> Vec<Self>;
}

pub trait IterExtent<N> {
    type PointIter: Iterator<Item = PointN<N>>;

    fn iter_extent(min: &PointN<N>, max: &PointN<N>) -> Self::PointIter;
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
