use crate::{Extent2i, Extent3i, IntegerPoint, Point, Point2i, Point3i, PointN};

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
    pub fn new(exponent: u8, min_multiple: PointN<N>) -> Self {
        let edge_length = 1 << exponent as i32;
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
