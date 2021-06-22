/// Rounds up to the nearest multiple of a power of 2.
#[inline]
pub fn round_up_multiple_of_pow2(x: i32, power: i32) -> i32 {
    (x + power - 1) & -power
}

/// Rounds down to the nearest multiple of a power of 2.
#[inline]
pub fn round_down_multiple_of_pow2(x: i32, power: i32) -> i32 {
    x & -power
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
    fn round_up_pow2() {
        assert_eq!(round_up_multiple_of_pow2(-4, 2), -4);
        assert_eq!(round_up_multiple_of_pow2(-3, 2), -2);
        assert_eq!(round_up_multiple_of_pow2(-2, 2), -2);
        assert_eq!(round_up_multiple_of_pow2(-1, 2), 0);
        assert_eq!(round_up_multiple_of_pow2(0, 2), 0);
        assert_eq!(round_up_multiple_of_pow2(1, 2), 2);
        assert_eq!(round_up_multiple_of_pow2(2, 2), 2);
        assert_eq!(round_up_multiple_of_pow2(3, 2), 4);
        assert_eq!(round_up_multiple_of_pow2(4, 2), 4);

        assert_eq!(round_up_multiple_of_pow2(-4, 4), -4);
        assert_eq!(round_up_multiple_of_pow2(-3, 4), 0);
        assert_eq!(round_up_multiple_of_pow2(-2, 4), 0);
        assert_eq!(round_up_multiple_of_pow2(-1, 4), 0);
        assert_eq!(round_up_multiple_of_pow2(0, 4), 0);
        assert_eq!(round_up_multiple_of_pow2(1, 4), 4);
        assert_eq!(round_up_multiple_of_pow2(2, 4), 4);
        assert_eq!(round_up_multiple_of_pow2(3, 4), 4);
        assert_eq!(round_up_multiple_of_pow2(4, 4), 4);
    }

    #[test]
    fn round_down_pow2() {
        assert_eq!(round_down_multiple_of_pow2(-4, 2), -4);
        assert_eq!(round_down_multiple_of_pow2(-3, 2), -4);
        assert_eq!(round_down_multiple_of_pow2(-2, 2), -2);
        assert_eq!(round_down_multiple_of_pow2(-1, 2), -2);
        assert_eq!(round_down_multiple_of_pow2(0, 2), 0);
        assert_eq!(round_down_multiple_of_pow2(1, 2), 0);
        assert_eq!(round_down_multiple_of_pow2(2, 2), 2);
        assert_eq!(round_down_multiple_of_pow2(3, 2), 2);
        assert_eq!(round_down_multiple_of_pow2(4, 2), 4);

        assert_eq!(round_down_multiple_of_pow2(-4, 4), -4);
        assert_eq!(round_down_multiple_of_pow2(-3, 4), -4);
        assert_eq!(round_down_multiple_of_pow2(-2, 4), -4);
        assert_eq!(round_down_multiple_of_pow2(-1, 4), -4);
        assert_eq!(round_down_multiple_of_pow2(0, 4), 0);
        assert_eq!(round_down_multiple_of_pow2(1, 4), 0);
        assert_eq!(round_down_multiple_of_pow2(2, 4), 0);
        assert_eq!(round_down_multiple_of_pow2(3, 4), 0);
        assert_eq!(round_down_multiple_of_pow2(4, 4), 4);
    }
}
