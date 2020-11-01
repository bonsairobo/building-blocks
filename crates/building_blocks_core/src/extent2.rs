use crate::{extent::IntegerExtent, Extent, ExtentN, Point2, PointN};

use core::ops::{Mul, Range};
use itertools::{iproduct, Product};

/// A 2-dimensional extent with scalar type `T`.
pub type Extent2<T> = ExtentN<[T; 2]>;
/// A 2-dimensional extent with scalar type `i32`.
pub type Extent2i = ExtentN<[i32; 2]>;
/// A 2-dimensional extent with scalar type `f32`.
pub type Extent2f = ExtentN<[f32; 2]>;

impl<T> Extent<[T; 2]> for Extent2<T>
where
    T: Copy + Mul<Output = T>,
{
    type VolumeType = T;

    fn volume(&self) -> T {
        self.shape.x() * self.shape.y()
    }
}

/// An iterator over all points in an `Extent2<T>`.
pub struct Extent2PointIter<T>
where
    Range<T>: Iterator<Item = T>,
{
    product_iter: Product<Range<T>, Range<T>>,
}

impl<T> Iterator for Extent2PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    type Item = Point2<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.product_iter.next().map(|(y, x)| PointN([x, y]))
    }
}

impl IntegerExtent<[i32; 2]> for Extent2i {
    type PointIter = Extent2PointIter<i32>;

    fn num_points(&self) -> usize {
        self.volume() as usize
    }

    fn iter_points(&self) -> Self::PointIter {
        let lub = self.least_upper_bound();

        Extent2PointIter {
            // iproduct is opposite of row-major order.
            product_iter: iproduct!(self.minimum.y()..lub.y(), self.minimum.x()..lub.x()),
        }
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn row_major_extent_iter() {
        let extent = Extent2i::from_min_and_shape(PointN([0, 0]), PointN([2, 2]));

        let points: Vec<_> = extent.iter_points().collect();

        assert_eq!(
            points,
            vec![
                PointN([0, 0]),
                PointN([1, 0]),
                PointN([0, 1]),
                PointN([1, 1]),
            ]
        );
    }
}
