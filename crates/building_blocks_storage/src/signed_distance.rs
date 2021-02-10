pub trait SignedDistance: Into<f32> {
    fn is_negative(self) -> bool;
}

impl SignedDistance for f32 {
    #[inline]
    fn is_negative(self) -> bool {
        self < 0.0
    }
}

/// A signed distance value in the range `[-1.0, 1.0]` with 8 bits of precision.
pub struct Sd8(pub i8);
/// A signed distance value in the range `[-1.0, 1.0]` with 16 bits of precision.
pub struct Sd16(pub i16);

impl Sd8 {
    const RESOLUTION: f32 = std::i8::MAX as f32;
    const PRECISION: f32 = 1.0 / Self::RESOLUTION;
}

impl Sd16 {
    const RESOLUTION: f32 = std::i16::MAX as f32;
    const PRECISION: f32 = 1.0 / Self::RESOLUTION;
}

impl From<Sd8> for f32 {
    fn from(s: Sd8) -> f32 {
        s.0 as f32 * Sd8::PRECISION
    }
}
impl From<f32> for Sd8 {
    fn from(s: f32) -> Self {
        Sd8((s.min(1.0).max(-1.0) / Self::RESOLUTION) as i8)
    }
}
impl SignedDistance for Sd8 {
    #[inline]
    fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl From<Sd16> for f32 {
    fn from(s: Sd16) -> f32 {
        s.0 as f32 * Sd16::PRECISION
    }
}
impl From<f32> for Sd16 {
    fn from(s: f32) -> Self {
        Sd16((s.min(1.0).max(-1.0) / Self::RESOLUTION) as i16)
    }
}
impl SignedDistance for Sd16 {
    #[inline]
    fn is_negative(self) -> bool {
        self.0 < 0
    }
}
