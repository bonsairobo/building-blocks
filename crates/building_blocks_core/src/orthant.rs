use crate::{point::point_traits::*, Extent2i, Extent3i, ExtentN, Point2i, Point3i, PointN};

use std::convert::TryFrom;

/// An extent for which, given some fixed power of 2 called P, satisfies:
/// - each component of the minimum is a multiple of P
/// - the shape is a cube with edge length P
///
/// Equivalently, this is the space covered by a single node of an octree centered at 0.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Orthant<N> {
    minimum: PointN<N>,
    edge_length: i32,
}

/// A 2D `Orthant`.
pub type Quadrant = Orthant<[i32; 2]>;

/// A 3D `Orthant`.
pub type Octant = Orthant<[i32; 3]>;

impl<N> Orthant<N>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Construct an `Orthant`. This ensures that the orthant is valid by constraining the input parameters to be:
    /// - an `exponent` for the power of 2 edge length
    /// - a `min_multiple` to multiply by the `edge_length` to get the minimum
    #[inline]
    pub fn new(exponent: i32, min_multiple: PointN<N>) -> Self {
        let edge_length = 1 << exponent;
        let minimum = min_multiple * edge_length;

        Self {
            minimum,
            edge_length,
        }
    }

    #[inline]
    pub fn new_unchecked(minimum: PointN<N>, edge_length: i32) -> Self {
        Self {
            minimum,
            edge_length,
        }
    }

    #[inline]
    pub fn minimum(&self) -> PointN<N> {
        self.minimum
    }

    #[inline]
    pub fn edge_length(&self) -> i32 {
        self.edge_length
    }

    #[inline]
    pub fn is_single_voxel(&self) -> bool {
        self.edge_length == 1
    }

    #[inline]
    pub fn exponent(&self) -> u8 {
        self.edge_length.trailing_zeros() as u8
    }
}

impl From<Quadrant> for Extent2i {
    #[inline]
    fn from(quad: Quadrant) -> Self {
        Extent2i::from_min_and_shape(quad.minimum, Point2i::fill(quad.edge_length))
    }
}

impl From<Octant> for Extent3i {
    #[inline]
    fn from(octant: Octant) -> Self {
        Extent3i::from_min_and_shape(octant.minimum, Point3i::fill(octant.edge_length))
    }
}

macro_rules! impl_try_from_extent_for_orthant {
    ($extent_ty:ty, $point_ty:ty, $orthant_ty:ty) => {
        impl TryFrom<$extent_ty> for $orthant_ty {
            type Error = ExtentIsNotOrthantError;

            #[inline]
            fn try_from(extent: $extent_ty) -> Result<Self, Self::Error> {
                let edge_length = extent.shape.x();

                if !extent.shape.dimensions_are_powers_of_2() {
                    Err(ExtentIsNotOrthantError::ShapeNotPow2)
                } else if !extent.shape.is_cube() {
                    Err(ExtentIsNotOrthantError::ShapeNotCube)
                } else if (extent.minimum % edge_length) != <$point_ty>::ZERO {
                    Err(ExtentIsNotOrthantError::MinimumNotPow2Multiple)
                } else {
                    Ok(<$orthant_ty>::new_unchecked(extent.minimum, edge_length))
                }
            }
        }
    };
}

impl_try_from_extent_for_orthant!(Extent2i, Point2i, Quadrant);
impl_try_from_extent_for_orthant!(Extent3i, Point3i, Octant);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtentIsNotOrthantError {
    MinimumNotPow2Multiple,
    ShapeNotPow2,
    ShapeNotCube,
}

/// Returns the smallest set of `Orthant`s required to cover `extent`. All `Orthant`s have the same shape, as determined by
/// `exponent`.
#[inline]
pub fn orthants_covering_extent<N>(
    extent: ExtentN<N>,
    exponent: i32,
) -> impl Iterator<Item = Orthant<N>>
where
    PointN<N>: IntegerPoint<N>,
{
    let scaled_extent =
        ExtentN::from_min_and_max(extent.minimum >> exponent, extent.max() >> exponent);

    scaled_extent
        .iter_points()
        .map(move |min_mul| Orthant::new(exponent, min_mul))
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
    fn orthants_cover_extent_exact() {
        let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([4, 4, 4]));

        let covering_octants: Vec<_> = orthants_covering_extent(extent, 2).collect();
        assert_eq!(covering_octants, vec![Orthant::new(2, PointN([0, 0, 0]))]);

        let covering_octants: Vec<_> = orthants_covering_extent(extent, 1).collect();
        assert_eq!(
            covering_octants,
            vec![
                Orthant::new(1, PointN([0, 0, 0])),
                Orthant::new(1, PointN([1, 0, 0])),
                Orthant::new(1, PointN([0, 1, 0])),
                Orthant::new(1, PointN([1, 1, 0])),
                Orthant::new(1, PointN([0, 0, 1])),
                Orthant::new(1, PointN([1, 0, 1])),
                Orthant::new(1, PointN([0, 1, 1])),
                Orthant::new(1, PointN([1, 1, 1])),
            ]
        );
    }

    #[test]
    fn orthants_cover_extent_inexact_negative() {
        let extent = Extent3i::from_min_and_shape(PointN([-3, -3, -3]), PointN([2, 2, 2]));

        let covering_octants: Vec<_> = orthants_covering_extent(extent, 2).collect();
        assert_eq!(
            covering_octants,
            vec![Orthant::new(2, PointN([-1, -1, -1]))]
        );

        let covering_octants: Vec<_> = orthants_covering_extent(extent, 1).collect();
        assert_eq!(
            covering_octants,
            vec![
                Orthant::new(1, PointN([-2, -2, -2])),
                Orthant::new(1, PointN([-1, -2, -2])),
                Orthant::new(1, PointN([-2, -1, -2])),
                Orthant::new(1, PointN([-1, -1, -2])),
                Orthant::new(1, PointN([-2, -2, -1])),
                Orthant::new(1, PointN([-1, -2, -1])),
                Orthant::new(1, PointN([-2, -1, -1])),
                Orthant::new(1, PointN([-1, -1, -1])),
            ]
        );
    }
}
