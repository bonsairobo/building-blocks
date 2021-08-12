use crate::{Point2i, Point3i};

use bitintr::{Pdep, Pext};
use std::fmt;

// ██████╗ ██████╗
// ╚════██╗██╔══██╗
//  █████╔╝██║  ██║
// ██╔═══╝ ██║  ██║
// ███████╗██████╔╝
// ╚══════╝╚═════╝

/// A Morton-encoded `Point2i`. Uses a `u64` to support the full set of `Point2i`s.
///
/// <https://en.wikipedia.org/wiki/Z-order_curve>
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct Morton2(pub u64);

impl Morton2 {
    // Only 32 bits can be set in each mask.
    const X_MASK: u64 =
        0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101;
    const Y_MASK: u64 =
        0b1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010;
}

impl fmt::Debug for Morton2 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {:b}", self.0, self.0)
    }
}

impl From<Point2i> for Morton2 {
    #[inline]
    fn from(p: Point2i) -> Self {
        Self(translate(p.x()).pdep(Self::X_MASK) | translate(p.y()).pdep(Self::Y_MASK))
    }
}

impl From<Morton2> for Point2i {
    #[inline]
    fn from(m: Morton2) -> Self {
        Self([
            untranslate(m.0.pext(Morton2::X_MASK)),
            untranslate(m.0.pext(Morton2::Y_MASK)),
        ])
    }
}

// ██████╗ ██████╗
// ╚════██╗██╔══██╗
//  █████╔╝██║  ██║
//  ╚═══██╗██║  ██║
// ██████╔╝██████╔╝
// ╚═════╝ ╚═════╝

/// A Morton-encoded `Point3i`. Uses a `u128` to support the full set of `Point3i`s.
///
/// <https://en.wikipedia.org/wiki/Z-order_curve>
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct Morton3(pub u128);

impl fmt::Debug for Morton3 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {:b}", self.0, self.0)
    }
}

// bitintr does not support u128, so we need to use two separate u64s and concatenate them.
impl Morton3 {
    // Only 21 bits can be set in each mask.
    const X_MASK: u64 =
        0b0001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001;
    const Y_MASK: u64 =
        0b0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010;
    const Z_MASK: u64 =
        0b0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100_1001_0010_0100;
}

impl From<Point3i> for Morton3 {
    #[inline]
    fn from(p: Point3i) -> Self {
        let x_t = translate(p.x());
        let y_t = translate(p.y());
        let z_t = translate(p.z());

        // PERF: might be better to use BEXTR instead of masking and shifting
        const LOW_21: u64 = (1 << 21) - 1;
        const HIGH_11: u64 = ((1 << 11) - 1) << 21;

        let x_t_low_21 = x_t & LOW_21;
        let y_t_low_21 = y_t & LOW_21;
        let z_t_low_21 = z_t & LOW_21;

        let morton_low_63 = x_t_low_21.pdep(Morton3::X_MASK)
            | y_t_low_21.pdep(Morton3::Y_MASK)
            | z_t_low_21.pdep(Morton3::Z_MASK);

        let x_t_high_11 = (x_t & HIGH_11) >> 21;
        let y_t_high_11 = (y_t & HIGH_11) >> 21;
        let z_t_high_11 = (z_t & HIGH_11) >> 21;

        let morton_high_33 = x_t_high_11.pdep(Morton3::X_MASK)
            | y_t_high_11.pdep(Morton3::Y_MASK)
            | z_t_high_11.pdep(Morton3::Z_MASK);

        Self(((morton_high_33 as u128) << 63) | morton_low_63 as u128)
    }
}

impl From<Morton3> for Point3i {
    #[inline]
    fn from(m: Morton3) -> Self {
        const LOW_63: u128 = (1 << 63) - 1;
        const HIGH_33: u128 = ((1 << 33) - 1) << 63;

        let m_low_63 = (m.0 & LOW_63) as u64;
        let m_high_33 = ((m.0 & HIGH_33) >> 63) as u64;

        let x_t_low_21 = m_low_63.pext(Morton3::X_MASK);
        let y_t_low_21 = m_low_63.pext(Morton3::Y_MASK);
        let z_t_low_21 = m_low_63.pext(Morton3::Z_MASK);

        let x_t_high_11 = m_high_33.pext(Morton3::X_MASK) << 21;
        let y_t_high_11 = m_high_33.pext(Morton3::Y_MASK) << 21;
        let z_t_high_11 = m_high_33.pext(Morton3::Z_MASK) << 21;

        Self([
            untranslate(x_t_high_11 | x_t_low_21),
            untranslate(y_t_high_11 | y_t_low_21),
            untranslate(z_t_high_11 | z_t_low_21),
        ])
    }
}

/// Send the supported range of i32 into the lower 32 bits of a u64 while preserving the total order.
#[inline]
fn translate(x: i32) -> u64 {
    x.wrapping_sub(i32::MIN) as u64
}

/// The inverse of `translate`.
#[inline]
fn untranslate(x: u64) -> i32 {
    (x as i32).wrapping_add(i32::MIN)
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
    fn limits_of_i32() {
        let min = PointN([i32::MIN; 3]);
        let max = PointN([i32::MAX; 3]);

        assert_eq!(Morton3::from(min), Morton3(0));
        assert_eq!(Morton3::from(max), Morton3((1 << 96) - 1));

        assert_eq!(min, Point3i::from(Morton3::from(min)));
        assert_eq!(max, Point3i::from(Morton3::from(max)));
    }

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
        let upper_bound = min + mortons.len() as u128;
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
