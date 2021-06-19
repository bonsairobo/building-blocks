use crate::{Point2i, Point3i};

use bitintr::{Pdep, Pext};
use std::fmt;

// ██████╗ ██████╗
// ╚════██╗██╔══██╗
//  █████╔╝██║  ██║
// ██╔═══╝ ██║  ██║
// ███████╗██████╔╝
// ╚══════╝╚═════╝

/// A Morton-encoded `Point2i`.
///
/// Since the encoding packs bits into a u64, and we have two i32s in a `Point2i`, all points can be encoded.
///
/// https://en.wikipedia.org/wiki/Z-order_curve
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct Morton2(pub u64);

impl Morton2 {
    // To pack 2 ints into a u64, we can use at most 32 bits per int.
    const MIN_SUPPORTED_INT: i32 = i32::MIN;
    const MAX_SUPPORTED_INT: i32 = i32::MAX;

    // Only 21 bits can be set in each mask.
    const X_MASK: u64 = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
    const Y_MASK: u64 = 0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010;

    /// Send the supported range of i32 into the lower 21 bits of a u64 while preserving the total order.
    #[inline]
    fn translate(x: i32) -> u64 {
        debug_assert!(x >= Self::MIN_SUPPORTED_INT);
        debug_assert!(x <= Self::MAX_SUPPORTED_INT);

        x.wrapping_sub(Self::MIN_SUPPORTED_INT) as u64
    }

    /// The inverse of `translate`.
    #[inline]
    fn untranslate(x: u64) -> i32 {
        (x as i32).wrapping_add(Self::MIN_SUPPORTED_INT)
    }
}

impl fmt::Debug for Morton2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {:b}", self.0, self.0)
    }
}

impl From<Point2i> for Morton2 {
    #[inline]
    fn from(p: Point2i) -> Self {
        Self(Self::translate(p.x()).pdep(Self::X_MASK) | Self::translate(p.y()).pdep(Self::Y_MASK))
    }
}

impl From<Morton2> for Point2i {
    #[inline]
    fn from(m: Morton2) -> Self {
        Self([
            Morton2::untranslate(m.0.pext(Morton2::X_MASK)),
            Morton2::untranslate(m.0.pext(Morton2::Y_MASK)),
        ])
    }
}

// ██████╗ ██████╗
// ╚════██╗██╔══██╗
//  █████╔╝██║  ██║
//  ╚═══██╗██║  ██║
// ██████╔╝██████╔╝
// ╚═════╝ ╚═════╝

/// A Morton-encoded `Point3i`.
///
/// Since the encoding packs bits into a u64, and we have three i32s in a `Point3i`, only 21 bits may be used per component.
/// This imposes a range on the components of `[Self::MIN_SUPPORTED_INT, Self::MAX_SUPPORTED_INT]`. Falling out of range will
/// cause a panic in debug mode only.
///
/// https://en.wikipedia.org/wiki/Z-order_curve
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct Morton3(pub u64);

impl fmt::Debug for Morton3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {:b}", self.0, self.0)
    }
}

impl Morton3 {
    // To pack 3 ints into a u64, we can use at most 21 bits per int.
    pub const MIN_SUPPORTED_INT: i32 = -(1 << 20);
    pub const MAX_SUPPORTED_INT: i32 = (1 << 20) - 1;

    // Only 21 bits can be set in each mask.
    pub const X_MASK: u64 =
        0b00010010_01001001_00100100_10010010_01001001_00100100_10010010_01001001;
    pub const Y_MASK: u64 =
        0b00100100_10010010_01001001_00100100_10010010_01001001_00100100_10010010;
    pub const Z_MASK: u64 =
        0b01001001_00100100_10010010_01001001_00100100_10010010_01001001_00100100;

    /// Send the supported range of i32 into the lower 21 bits of a u64 while preserving the total order.
    #[inline]
    fn translate(x: i32) -> u64 {
        debug_assert!(x >= Morton3::MIN_SUPPORTED_INT);
        debug_assert!(x <= Morton3::MAX_SUPPORTED_INT);

        x.wrapping_sub(Morton3::MIN_SUPPORTED_INT) as u64
    }

    /// The inverse of `translate`.
    #[inline]
    fn untranslate(x: u64) -> i32 {
        (x as i32).wrapping_add(Morton3::MIN_SUPPORTED_INT)
    }
}

impl From<Point3i> for Morton3 {
    #[inline]
    fn from(p: Point3i) -> Self {
        Self(
            Self::translate(p.x()).pdep(Morton3::X_MASK)
                | Self::translate(p.y()).pdep(Morton3::Y_MASK)
                | Self::translate(p.z()).pdep(Morton3::Z_MASK),
        )
    }
}

impl From<Morton3> for Point3i {
    #[inline]
    fn from(m: Morton3) -> Self {
        Self([
            Morton3::untranslate(m.0.pext(Morton3::X_MASK)),
            Morton3::untranslate(m.0.pext(Morton3::Y_MASK)),
            Morton3::untranslate(m.0.pext(Morton3::Z_MASK)),
        ])
    }
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
    use crate::PointN;

    #[test]
    fn octants_are_contiguous_in_morton_space() {
        let octant_mins = [
            [-2, -2, -2],
            [0, -2, -2],
            [-2, 0, -2],
            [0, 0, -2],
            [-2, -2, 0],
            [0, -2, 0],
            [-2, 0, 0],
            [0, 0, 0],
        ];

        for &octant_min in octant_mins.iter() {
            let octant_points: Vec<_> = Z_OFFSETS
                .iter()
                .cloned()
                .map(|offset| PointN(octant_min) + PointN(offset))
                .collect();

            // Decode is inverse of encode.
            for &p in octant_points.iter() {
                assert_eq!(p, Point3i::from(Morton3::from(p)));
            }

            let octant_mortons: Vec<_> = octant_points
                .into_iter()
                .map(|p| Morton3::from(p))
                .collect();

            assert!(mortons_are_contiguous(&octant_mortons));
        }
    }

    fn mortons_are_contiguous(mortons: &[Morton3]) -> bool {
        let min = mortons[0].0;
        let upper_bound = min + mortons.len() as u64;
        for (i, expected) in (min..upper_bound).enumerate() {
            if mortons[i] != Morton3(expected) {
                return false;
            }
        }

        true
    }

    const Z_OFFSETS: [[i32; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
}
