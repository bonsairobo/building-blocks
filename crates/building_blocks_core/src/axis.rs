use crate::{Point2i, Point3i, PointN};

/// Either the X or Y axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis2 {
    X = 0,
    Y = 1,
}

impl Axis2 {
    /// The index for a point's component on this axis.
    #[inline]
    pub fn index(&self) -> usize {
        *self as usize
    }

    #[inline]
    pub fn get_unit_vector(&self) -> Point2i {
        match self {
            Axis2::X => PointN([1, 0]),
            Axis2::Y => PointN([0, 1]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedAxis2 {
    pub sign: i32,
    pub axis: Axis2,
}

impl SignedAxis2 {
    #[inline]
    pub fn new(sign: i32, axis: Axis2) -> Self {
        Self { sign, axis }
    }

    #[inline]
    pub fn get_vector(&self) -> Point2i {
        self.axis.get_unit_vector() * self.sign
    }

    #[inline]
    pub fn from_vector(v: Point2i) -> Option<Self> {
        match v {
            PointN([x, 0]) => Some(SignedAxis2::new(x, Axis2::X)),
            PointN([0, y]) => Some(SignedAxis2::new(y, Axis2::Y)),
            _ => None,
        }
    }
}

/// Either the X, Y, or Z axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis3 {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis3 {
    /// The index for a point's component on this axis.
    #[inline]
    pub fn index(&self) -> usize {
        *self as usize
    }

    #[inline]
    pub const fn get_unit_vector(&self) -> Point3i {
        match self {
            Axis3::X => PointN([1, 0, 0]),
            Axis3::Y => PointN([0, 1, 0]),
            Axis3::Z => PointN([0, 0, 1]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis3Permutation {
    // Even permutations
    Xyz,
    Zxy,
    Yzx,
    // Odd permutations
    Zyx,
    Xzy,
    Yxz,
}

impl Axis3Permutation {
    #[inline]
    pub const fn even_with_normal_axis(axis: Axis3) -> Self {
        match axis {
            Axis3::X => Axis3Permutation::Xyz,
            Axis3::Y => Axis3Permutation::Yzx,
            Axis3::Z => Axis3Permutation::Zxy,
        }
    }

    #[inline]
    pub const fn odd_with_normal_axis(axis: Axis3) -> Self {
        match axis {
            Axis3::X => Axis3Permutation::Xzy,
            Axis3::Y => Axis3Permutation::Yxz,
            Axis3::Z => Axis3Permutation::Zyx,
        }
    }

    #[inline]
    pub const fn sign(&self) -> i32 {
        match self {
            Axis3Permutation::Xyz => 1,
            Axis3Permutation::Zxy => 1,
            Axis3Permutation::Yzx => 1,
            Axis3Permutation::Zyx => -1,
            Axis3Permutation::Xzy => -1,
            Axis3Permutation::Yxz => -1,
        }
    }

    #[inline]
    pub const fn axes(&self) -> [Axis3; 3] {
        match self {
            Axis3Permutation::Xyz => [Axis3::X, Axis3::Y, Axis3::Z],
            Axis3Permutation::Zxy => [Axis3::Z, Axis3::X, Axis3::Y],
            Axis3Permutation::Yzx => [Axis3::Y, Axis3::Z, Axis3::X],
            Axis3Permutation::Zyx => [Axis3::Z, Axis3::Y, Axis3::X],
            Axis3Permutation::Xzy => [Axis3::X, Axis3::Z, Axis3::Y],
            Axis3Permutation::Yxz => [Axis3::Y, Axis3::X, Axis3::Z],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedAxis3 {
    pub sign: i32,
    pub axis: Axis3,
}

impl SignedAxis3 {
    #[inline]
    pub const fn new(sign: i32, axis: Axis3) -> Self {
        Self { sign, axis }
    }

    #[inline]
    pub fn get_vector(&self) -> Point3i {
        self.axis.get_unit_vector() * self.sign
    }

    #[inline]
    pub const fn from_vector(v: Point3i) -> Option<Self> {
        match v {
            PointN([x, 0, 0]) => Some(SignedAxis3::new(x, Axis3::X)),
            PointN([0, y, 0]) => Some(SignedAxis3::new(y, Axis3::Y)),
            PointN([0, 0, z]) => Some(SignedAxis3::new(z, Axis3::Z)),
            _ => None,
        }
    }
}
