use crate::{Extent3i, Point, Point3i};

/// An extent for which, given some fixed power of 2 called P, satisfies:
/// - each component of the minimum is a multiple of P
/// - the shape is a cube with edge length P
///
/// Equivalently, this is the space covered by a single node of an octree centered at 0.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Octant {
    minimum: Point3i,
    edge_length: i32,
}

impl Octant {
    /// Construct an `Octant`. This ensures that the octant is valid by constraining the input parameters to be:
    /// - an `exponent` for the power of 2 edge length
    /// - a `min_multiple` to multiply by the `edge_length` to get the minimum
    #[inline]
    pub fn new(exponent: u8, min_multiple: Point3i) -> Self {
        let edge_length = 1 << exponent as i32;
        let minimum = min_multiple * edge_length;

        Self {
            minimum,
            edge_length,
        }
    }

    #[inline]
    pub fn new_unchecked(minimum: Point3i, edge_length: i32) -> Self {
        Self {
            minimum,
            edge_length,
        }
    }

    #[inline]
    pub fn minimum(&self) -> Point3i {
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

impl From<Octant> for Extent3i {
    #[inline]
    fn from(octant: Octant) -> Self {
        Extent3i::from_min_and_shape(octant.minimum, Point3i::fill(octant.edge_length))
    }
}
