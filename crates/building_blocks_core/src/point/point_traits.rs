use crate::PointN;

use core::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub, SubAssign,
};
use num::Zero;

/// A trait that bundles op traits that all `PointN<N>` (and its components) should have.
pub trait Point:
    'static
    + Abs
    + Add<Output = Self>
    + AddAssign
    + Bounded
    + ConstZero
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

    fn fill(value: <Self as Point>::Scalar) -> Self;

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
    type Scalar;

    /// Returns the point after applying `f` component-wise.
    fn map_components_unary(&self, f: impl Fn(Self::Scalar) -> Self::Scalar) -> Self;

    /// Returns the point after applying `f` component-wise to both `self` and `other` in parallel.
    fn map_components_binary(
        &self,
        other: &Self,
        f: impl Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self;
}

pub trait MinMaxComponent {
    type Scalar;

    fn min_component(&self) -> Self::Scalar;
    fn max_component(&self) -> Self::Scalar;
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
    BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + BitAnd<i32, Output = Self>
    + BitOr<i32, Output = Self>
    + BitXor<i32, Output = Self>
    + Eq
    + IntegerDiv
    + IterExtent<N>
    + LatticeOrder
    + Neighborhoods
    + Not<Output = Self>
    + Point<Scalar = i32>
    + Rem<Self, Output = Self>
    + Shl<Self, Output = Self>
    + Shr<Self, Output = Self>
    + Rem<i32, Output = Self>
    + Shl<i32, Output = Self>
    + Shr<i32, Output = Self>
{
    /// Returns `true` iff all dimensions are powers of 2.
    fn dimensions_are_powers_of_2(&self) -> bool;

    /// Returns `true` iff all dimensions are equal.
    fn is_cube(&self) -> bool;
}

pub trait IntegerDiv {
    fn vector_div_floor(&self, rhs: &Self) -> Self;

    fn scalar_div_floor(&self, rhs: i32) -> Self;

    fn vector_div_ceil(&self, rhs: &Self) -> Self;

    fn scalar_div_ceil(&self, rhs: i32) -> Self;
}

pub trait LatticeOrder {
    /// Component-wise maximum.
    fn join(&self, other: &Self) -> Self;

    /// Component-wise minimum.
    fn meet(&self, other: &Self) -> Self;
}

macro_rules! impl_unary_ops {
    ($t:ty, $scalar:ty) => {
        impl Mul<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| rhs * c)
            }
        }

        impl Mul<$t> for $scalar {
            type Output = $t;

            #[inline]
            fn mul(self, rhs: $t) -> $t {
                rhs * self
            }
        }
    };
}

macro_rules! impl_binary_ops {
    ($t:ty, $scalar:ty) => {
        impl LatticeOrder for $t {
            #[inline]
            fn join(&self, other: &Self) -> Self {
                self.map_components_binary(other, <$scalar>::max)
            }

            #[inline]
            fn meet(&self, other: &Self) -> Self {
                self.map_components_binary(other, <$scalar>::min)
            }
        }

        impl Mul<Self> for $t {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 * c2)
            }
        }
    };
}

macro_rules! impl_unary_float_ops {
    ($t:ty) => {
        impl $t {
            #[inline]
            pub fn round(&self) -> Self {
                self.map_components_unary(|c| c.round())
            }

            #[inline]
            pub fn floor(&self) -> Self {
                self.map_components_unary(|c| c.floor())
            }

            #[inline]
            pub fn ceil(&self) -> Self {
                self.map_components_unary(|c| c.ceil())
            }

            #[inline]
            pub fn fract(&self) -> Self {
                self.map_components_unary(|c| c.fract())
            }
        }
    };
}

macro_rules! impl_unary_integer_ops {
    ($t:ty, $scalar:ty) => {
        impl BitAnd<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c & rhs)
            }
        }

        impl BitOr<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c | rhs)
            }
        }

        impl BitXor<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c ^ rhs)
            }
        }

        impl Not for $t {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                self.map_components_unary(|c| !c)
            }
        }

        impl Rem<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c % rhs)
            }
        }

        impl Shl<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c << rhs)
            }
        }

        impl Shr<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c >> rhs)
            }
        }
    };
}

macro_rules! impl_binary_integer_ops {
    ($t:ty) => {
        impl BitAnd<Self> for $t {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 & c2)
            }
        }

        impl BitOr<Self> for $t {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 | c2)
            }
        }

        impl BitXor<Self> for $t {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 ^ c2)
            }
        }

        impl Rem<Self> for $t {
            type Output = Self;

            #[inline]
            fn rem(self, other: Self) -> Self {
                self.map_components_binary(&other, |c1, c2| c1 % c2)
            }
        }

        impl Shl<Self> for $t {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 << c2)
            }
        }

        impl Shr<Self> for $t {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 >> c2)
            }
        }
    };
}

macro_rules! impl_float_div {
    ($t:ty, $scalar:ty) => {
        impl Div<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn div(self, rhs: $scalar) -> Self {
                self.map_components_unary(|c| c / rhs)
            }
        }

        impl Div<Self> for $t {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self {
                self.map_components_binary(&rhs, |c1, c2| c1 / c2)
            }
        }
    };
}

macro_rules! impl_integer_div {
    ($t:ty, $scalar:ty) => {
        // Use specialized implementation for integers because the default Div impl rounds towards zero, which is not what we
        // want.
        impl Div<$scalar> for $t {
            type Output = Self;

            #[inline]
            fn div(self, rhs: $scalar) -> Self {
                self.scalar_div_floor(rhs)
            }
        }

        // Use specialized implementation for integers because the default Div impl rounds towards zero,
        // which is not what we want.
        impl Div<Self> for $t {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self {
                self.vector_div_floor(&rhs)
            }
        }

        impl IntegerDiv for $t {
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
    };
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
pub trait ConstZero: Copy {
    const ZERO: Self;
}

// `One` trait doesn't allow associated constants for one because of bignums.
pub trait ConstOne: Copy {
    const ONE: Self;
}

impl ConstZero for i32 {
    const ZERO: i32 = 0;
}
impl ConstOne for i32 {
    const ONE: i32 = 1;
}

impl ConstZero for f32 {
    const ZERO: f32 = 0.0;
}
impl ConstOne for f32 {
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
