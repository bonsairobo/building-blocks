use crate::{extent::IntegerExtent, Extent, ExtentN, Point3, PointN};

use core::ops::{Mul, Range};
use itertools::{iproduct, ConsTuples, Product};

/// A 3-dimensional extent with scalar type `T`.
pub type Extent3<T> = ExtentN<[T; 3]>;
/// A 3-dimensional extent with scalar type `i32`.
pub type Extent3i = ExtentN<[i32; 3]>;

impl<T> Extent<[T; 3]> for Extent3<T>
where
    T: Copy + Mul<Output = T>,
{
    type VolumeType = T;

    fn volume(&self) -> T {
        self.shape.x() * self.shape.y() * self.shape.z()
    }
}

/// An iterator over all points in an `Extent3<T>`.
pub struct Extent3PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    product_iter: ConsTuples<RangeProduct3<T>, ((T, T), T)>,
}

type RangeProduct2<T> = Product<Range<T>, Range<T>>;
type RangeProduct3<T> = Product<RangeProduct2<T>, Range<T>>;

impl<T> Iterator for Extent3PointIter<T>
where
    T: Clone,
    Range<T>: Iterator<Item = T>,
{
    type Item = Point3<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.product_iter.next().map(|(z, y, x)| PointN([x, y, z]))
    }
}

impl IntegerExtent<[i32; 3]> for Extent3i {
    type PointIter = Extent3PointIter<i32>;

    fn num_points(&self) -> usize {
        self.volume() as usize
    }

    #[inline(always)]
    fn iter_points(&self) -> Self::PointIter {
        let lub = self.least_upper_bound();

        Extent3PointIter {
            // iproduct is opposite of row-major order.
            product_iter: iproduct!(
                self.minimum.z()..lub.z(),
                self.minimum.y()..lub.y(),
                self.minimum.x()..lub.x()
            ),
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
        let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([2, 2, 2]));

        let points: Vec<_> = extent.iter_points().collect();

        assert_eq!(
            points,
            vec![
                PointN([0, 0, 0]),
                PointN([1, 0, 0]),
                PointN([0, 1, 0]),
                PointN([1, 1, 0]),
                PointN([0, 0, 1]),
                PointN([1, 0, 1]),
                PointN([0, 1, 1]),
                PointN([1, 1, 1]),
            ]
        );
    }
}
