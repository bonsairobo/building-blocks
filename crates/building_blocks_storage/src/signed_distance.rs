pub trait SignedDistance: Copy + Into<f32> {
    const ZERO: Self;
    fn is_negative(self) -> bool;
}

impl SignedDistance for f32 {
    const ZERO: Self = 0.0;

    fn is_negative(self) -> bool {
        self < 0.0
    }
}
