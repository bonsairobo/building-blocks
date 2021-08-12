#[macro_use]
pub mod point_traits;

#[cfg(feature = "glam")]
mod glam_conversions;
#[cfg(feature = "mint")]
mod mint_conversions;
#[cfg(feature = "nalgebra")]
mod nalgebra_conversions;
#[cfg(feature = "sdfu")]
mod sdfu_integration;
#[cfg(feature = "vox-format")]
mod vox_format_conversions;

mod point2;
mod point3;

pub use point2::*;
pub use point3::*;

use point_traits::*;

use bytemuck::{Pod, Zeroable};
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};
use num::{Signed, Zero};
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
/// # use building_blocks_core::prelude::*;
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

unsafe impl<N> Zeroable for PointN<N> where N: Zeroable {}
unsafe impl<N> Pod for PointN<N> where N: Pod {}

impl<N> PointN<N>
where
    Self: MapComponents,
{
    #[inline]
    pub fn signum(self) -> Self
    where
        <Self as MapComponents>::Scalar: Signed,
    {
        self.map_components_unary(|c| c.signum())
    }
}

impl<N> Abs for PointN<N>
where
    Self: MapComponents,
    <Self as MapComponents>::Scalar: Signed,
{
    #[inline]
    fn abs(self) -> Self {
        self.map_components_unary(|c| c.abs())
    }
}

impl<N> Neg for PointN<N>
where
    Self: Copy + Sub<Output = Self> + Zero,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl<N, T> Add for PointN<N>
where
    Self: MapComponents<Scalar = T>,
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.map_components_binary(rhs, |c1, c2| c1 + c2)
    }
}

impl<N, T> Sub for PointN<N>
where
    Self: MapComponents<Scalar = T>,
    T: Sub<Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.map_components_binary(rhs, |c1, c2| c1 - c2)
    }
}

impl<N> AddAssign for PointN<N>
where
    Self: Copy + Add<Output = Self>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<N> SubAssign for PointN<N>
where
    Self: Copy + Sub<Output = Self>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<N> Zero for PointN<N>
where
    Self: Point + ConstZero,
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
