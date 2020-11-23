use super::*;

impl<T> From<mint::Point2<T>> for Point2<T> {
    #[inline]
    fn from(p: mint::Point2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl<T> From<Point2<T>> for mint::Point2<T>
where
    T: Clone,
{
    #[inline]
    fn from(p: Point2<T>) -> Self {
        mint::Point2::from_slice(&p.0)
    }
}

impl<T> From<mint::Vector2<T>> for Point2<T> {
    #[inline]
    fn from(p: mint::Vector2<T>) -> Self {
        PointN([p.x, p.y])
    }
}

impl<T> From<Point2<T>> for mint::Vector2<T>
where
    T: Clone,
{
    #[inline]
    fn from(p: Point2<T>) -> Self {
        mint::Vector2::from_slice(&p.0)
    }
}

impl<T> From<mint::Point3<T>> for Point3<T> {
    #[inline]
    fn from(p: mint::Point3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl<T> From<Point3<T>> for mint::Point3<T>
where
    T: Clone,
{
    #[inline]
    fn from(p: Point3<T>) -> Self {
        mint::Point3::from_slice(&p.0)
    }
}

impl<T> From<mint::Vector3<T>> for Point3<T> {
    #[inline]
    fn from(p: mint::Vector3<T>) -> Self {
        PointN([p.x, p.y, p.z])
    }
}

impl<T> From<Point3<T>> for mint::Vector3<T>
where
    T: Clone,
{
    #[inline]
    fn from(p: Point3<T>) -> Self {
        mint::Vector3::from_slice(&p.0)
    }
}
