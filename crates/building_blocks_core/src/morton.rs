use crate::{Point2i, Point3i};

use morton_encoding::{morton_decode, morton_encode};
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

impl fmt::Debug for Morton2 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {:b}", self.0, self.0)
    }
}

impl From<Point2i> for Morton2 {
    #[inline]
    fn from(p: Point2i) -> Self {
        Self(morton_encode([
            translate(p.y()) as u32,
            translate(p.x()) as u32,
        ]))
    }
}

impl From<Morton2> for Point2i {
    #[inline]
    fn from(m: Morton2) -> Self {
        let yx: [u32; 2] = morton_decode(m.0);
        Self([untranslate(yx[1] as i32), untranslate(yx[0] as i32)])
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

impl From<Point3i> for Morton3 {
    #[inline]
    fn from(p: Point3i) -> Self {
        Self(morton_encode([
            translate(p.z()) as u32,
            translate(p.y()) as u32,
            translate(p.x()) as u32,
        ]))
    }
}

impl From<Morton3> for Point3i {
    #[inline]
    fn from(m: Morton3) -> Self {
        let zyx: [u32; 3] = morton_decode(m.0);
        Self([
            untranslate(zyx[2] as i32),
            untranslate(zyx[1] as i32),
            untranslate(zyx[0] as i32),
        ])
    }
}

/// Send the supported range of i32 into the lower 32 bits of a u64 while preserving the total order.
#[inline]
fn translate(x: i32) -> i32 {
    x.wrapping_sub(i32::MIN)
}

/// The inverse of `translate`.
#[inline]
fn untranslate(x: i32) -> i32 {
    x.wrapping_add(i32::MIN)
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
