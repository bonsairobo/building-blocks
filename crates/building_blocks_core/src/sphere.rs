use crate::prelude::{Distance, ExtentN, FloatPoint, Point, PointN};

pub struct Sphere<Nf> {
    pub center: PointN<Nf>,
    pub radius: f32,
}

pub type Sphere2 = Sphere<[f32; 2]>;
pub type Sphere3 = Sphere<[f32; 3]>;

impl<N> Clone for Sphere<N>
where
    PointN<N>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            center: self.center.clone(),
            radius: self.radius,
        }
    }
}
impl<N> Copy for Sphere<N> where PointN<N>: Copy {}

impl<Nf> Sphere<Nf>
where
    PointN<Nf>: FloatPoint,
{
    pub fn contains(&self, other: &Self) -> bool {
        let dist = self.center.l2_distance_squared(other.center).sqrt();
        dist + other.radius < self.radius
    }

    pub fn intersects(&self, other: &Self) -> bool {
        let dist = self.center.l2_distance_squared(other.center).sqrt();
        dist - other.radius < self.radius
    }

    pub fn aabb(&self) -> ExtentN<Nf> {
        ExtentN::from_min_and_shape(PointN::fill(-self.radius), PointN::fill(2.0 * self.radius))
            + self.center
    }
}
