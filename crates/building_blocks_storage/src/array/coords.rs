use building_blocks_core::{ConstZero, PointN};

use core::ops::{Add, AddAssign, Deref, Mul, Sub, SubAssign};
use num::Zero;

/// Map-local coordinates.
///
/// Most commonly, you will index a lattice map with a `PointN<N>`, which is assumed to be in global
/// coordinates. `Local<N>` only applies to lattice maps where a point must first be translated from
/// global coordinates into map-local coordinates before indexing with `Get<Local<N>>`.
#[derive(Debug, Eq, PartialEq)]
pub struct Local<N>(pub PointN<N>);

/// Map-local coordinates, wrapping a `Point2i`.
pub type Local2i = Local<[i32; 2]>;
/// Map-local coordinates, wrapping a `Point3i`.
pub type Local3i = Local<[i32; 3]>;

impl<N> Clone for Local<N>
where
    PointN<N>: Clone,
{
    fn clone(&self) -> Self {
        Local(self.0.clone())
    }
}
impl<N> Copy for Local<N> where PointN<N>: Copy {}

impl<N> Local<N> {
    /// Wraps all of the `points` using the `Local` constructor.
    #[inline]
    pub fn localize_points_slice(points: &[PointN<N>]) -> Vec<Local<N>>
    where
        PointN<N>: Clone,
    {
        points.iter().cloned().map(Local).collect()
    }

    /// Wraps all of the `points` using the `Local` constructor.
    #[inline]
    pub fn localize_points_array<const LEN: usize>(points: &[PointN<N>; LEN]) -> [Local<N>; LEN]
    where
        PointN<N>: ConstZero,
    {
        let mut locals = [Local(PointN::ZERO); LEN];
        for (l, p) in locals.iter_mut().zip(points.iter()) {
            *l = Local(*p);
        }

        locals
    }
}

impl<N> Deref for Local<N> {
    type Target = PointN<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The most efficient coordinates for slice-backed lattice maps. A single number that translates directly to a slice offset.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Stride(pub usize);

impl Zero for Stride {
    #[inline]
    fn zero() -> Self {
        Stride(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Add for Stride {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        // Wraps for negative point offsets.
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Stride {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        // Wraps for negative point offsets.
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul<usize> for Stride {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: usize) -> Self::Output {
        Self(self.0.wrapping_mul(rhs))
    }
}

impl AddAssign for Stride {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Stride {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
