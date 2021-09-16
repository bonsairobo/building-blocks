use bytemuck::{Pod, Zeroable};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait SignedDistance: Into<f32> {
    fn is_negative(&self) -> bool;
}

impl SignedDistance for f32 {
    #[inline]
    fn is_negative(&self) -> bool {
        *self < 0.0
    }
}

/// A signed distance value in the range `[-1.0, 1.0]` with 8 bits of precision.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sd8(pub i8);
/// A signed distance value in the range `[-1.0, 1.0]` with 16 bits of precision.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sd16(pub i16);

unsafe impl Zeroable for Sd8 {}
unsafe impl Pod for Sd8 {}

unsafe impl Zeroable for Sd16 {}
unsafe impl Pod for Sd16 {}

impl Sd8 {
    pub const RESOLUTION: f32 = std::i8::MAX as f32;
    pub const PRECISION: f32 = 1.0 / Self::RESOLUTION;
    pub const NEG_ONE: Self = Self(-std::i8::MAX);
    pub const ONE: Self = Self(std::i8::MAX);
}

impl Sd16 {
    pub const RESOLUTION: f32 = std::i16::MAX as f32;
    pub const PRECISION: f32 = 1.0 / Self::RESOLUTION;
    pub const NEG_ONE: Self = Self(-std::i16::MAX);
    pub const ONE: Self = Self(std::i16::MAX);
}

impl From<Sd8> for f32 {
    #[inline]
    fn from(s: Sd8) -> f32 {
        s.0 as f32 * Sd8::PRECISION
    }
}
impl From<f32> for Sd8 {
    #[inline]
    fn from(s: f32) -> Self {
        Sd8((Self::RESOLUTION * s.min(1.0).max(-1.0)) as i8)
    }
}
impl SignedDistance for Sd8 {
    #[inline]
    fn is_negative(&self) -> bool {
        self.0 < 0
    }
}

impl From<Sd16> for f32 {
    #[inline]
    fn from(s: Sd16) -> f32 {
        s.0 as f32 * Sd16::PRECISION
    }
}
impl From<f32> for Sd16 {
    #[inline]
    fn from(s: f32) -> Self {
        Sd16((Self::RESOLUTION * s.min(1.0).max(-1.0)) as i16)
    }
}
impl SignedDistance for Sd16 {
    #[inline]
    fn is_negative(&self) -> bool {
        self.0 < 0
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd8_boundary_conversions() {
        assert_eq!(-1.0, f32::from(Sd8::NEG_ONE));
        assert_eq!(1.0, f32::from(Sd8::ONE));
        assert_eq!(0.0, f32::from(Sd8(0)));

        assert_eq!(Sd8::NEG_ONE, Sd8::from(-1.0));
        assert_eq!(Sd8::ONE, Sd8::from(1.0));
        assert_eq!(Sd8(0), Sd8::from(0.0));
    }

    #[test]
    fn sd16_boundary_conversions() {
        assert_eq!(-1.0, f32::from(Sd16::NEG_ONE));
        assert_eq!(1.0, f32::from(Sd16::ONE));
        assert_eq!(0.0, f32::from(Sd16(0)));

        assert_eq!(Sd16::NEG_ONE, Sd16::from(-1.0));
        assert_eq!(Sd16::ONE, Sd16::from(1.0));
        assert_eq!(Sd16(0), Sd16::from(0.0));
    }
}
