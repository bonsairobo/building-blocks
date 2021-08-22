use crate::PointN;

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
