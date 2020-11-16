/// Either the X or Y axis.
#[derive(Clone, Copy)]
pub enum Axis2 {
    X = 0,
    Y = 1,
}

impl Axis2 {
    /// The index for a point's component on this axis.
    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// Either the X, Y, or Z axis.
#[derive(Clone, Copy)]
pub enum Axis3 {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis3 {
    /// The index for a point's component on this axis.
    pub fn index(&self) -> usize {
        *self as usize
    }
}

pub enum Axis3Permutation {
    // Even permutations
    XYZ,
    ZXY,
    YZX,
    // Odd permutations
    ZYX,
    XZY,
    YXZ,
}

impl Axis3Permutation {
    pub fn even_with_normal_axis(axis: Axis3) -> Self {
        match axis {
            Axis3::X => Axis3Permutation::XYZ,
            Axis3::Y => Axis3Permutation::YZX,
            Axis3::Z => Axis3Permutation::ZXY,
        }
    }

    pub fn odd_with_normal_axis(axis: Axis3) -> Self {
        match axis {
            Axis3::X => Axis3Permutation::XZY,
            Axis3::Y => Axis3Permutation::YXZ,
            Axis3::Z => Axis3Permutation::ZYX,
        }
    }

    pub fn sign(&self) -> i32 {
        match self {
            Axis3Permutation::XYZ => 1,
            Axis3Permutation::ZXY => 1,
            Axis3Permutation::YZX => 1,
            Axis3Permutation::ZYX => -1,
            Axis3Permutation::XZY => -1,
            Axis3Permutation::YXZ => -1,
        }
    }

    pub fn axes(&self) -> [Axis3; 3] {
        match self {
            Axis3Permutation::XYZ => [Axis3::X, Axis3::Y, Axis3::Z],
            Axis3Permutation::ZXY => [Axis3::Z, Axis3::X, Axis3::Y],
            Axis3Permutation::YZX => [Axis3::Y, Axis3::Z, Axis3::X],
            Axis3Permutation::ZYX => [Axis3::Z, Axis3::Y, Axis3::X],
            Axis3Permutation::XZY => [Axis3::X, Axis3::Z, Axis3::Y],
            Axis3Permutation::YXZ => [Axis3::Y, Axis3::X, Axis3::Z],
        }
    }
}
