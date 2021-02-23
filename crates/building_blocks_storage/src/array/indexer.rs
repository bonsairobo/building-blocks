use crate::{Local, Stride};

use building_blocks_core::{ConstZero, ExtentN, IntegerPoint, PointN};

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each_point_and_stride_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_min: PointN<N>,
        iter_shape: PointN<N>,
        f: impl FnMut(PointN<N>, Stride),
    );

    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &ExtentN<N>,
        array1_extent: &ExtentN<N>,
        array2_extent: &ExtentN<N>,
        f: impl FnMut(Stride, Stride),
    );

    #[inline]
    fn for_each_point_and_stride_local_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_shape: PointN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: Copy,
    {
        Self::for_each_point_and_stride_unchecked(
            array_shape,
            index_min,
            index_min.0,
            iter_shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_local(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(&ExtentN::from_min_and_shape(
            PointN::ZERO,
            array_extent.shape,
        ));

        Self::for_each_point_and_stride_local_unchecked(
            array_extent.shape,
            Local(iter_extent.minimum),
            iter_extent.shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_global_unchecked(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Translate to local coordinates.
        let index_min = iter_extent.minimum - array_extent.minimum;

        Self::for_each_point_and_stride_unchecked(
            array_extent.shape,
            Local(index_min),
            iter_extent.minimum,
            iter_extent.shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_global(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(array_extent);

        Self::for_each_point_and_stride_global_unchecked(array_extent, &iter_extent, f);
    }

    #[inline]
    fn strides_from_local_points(shape: PointN<N>, points: &[Local<N>], strides: &mut [Stride])
    where
        PointN<N>: Copy,
    {
        for (i, p) in points.iter().enumerate() {
            strides[i] = Self::stride_from_local_point(shape, *p);
        }
    }
}
