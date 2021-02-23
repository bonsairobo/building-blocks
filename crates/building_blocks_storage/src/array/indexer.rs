use crate::{Local, Stride};

use building_blocks_core::prelude::*;

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each_point_and_stride_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_extent: ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    );

    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &ExtentN<N>,
        array1_extent: &ExtentN<N>,
        array2_extent: &ExtentN<N>,
        f: impl FnMut(Stride, Stride),
    );

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

pub struct ArrayExtentVisitor<N> {
    /// Shape of the array being indexed.
    array_shape: PointN<N>,
    /// Array-local minimum where we start indexing.
    index_min: Local<N>,
    /// Extent of the iteration coordinates.
    iter_extent: ExtentN<N>,
}

impl<N> ArrayExtentVisitor<N>
where
    PointN<N>: IntegerPoint<N>,
{
    #[inline]
    pub fn new_local_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_shape: PointN<N>,
    ) -> Self {
        Self {
            array_shape,
            index_min,
            iter_extent: ExtentN::from_min_and_shape(index_min.0, iter_shape),
        }
    }

    #[inline]
    pub fn new_local(array_shape: PointN<N>, iter_extent: &ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent =
            iter_extent.intersection(&ExtentN::from_min_and_shape(PointN::ZERO, array_shape));

        Self::new_local_unchecked(array_shape, Local(iter_extent.minimum), iter_extent.shape)
    }

    #[inline]
    pub fn new_global_unchecked(array_extent: &ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Translate to local coordinates.
        let index_min = Local(iter_extent.minimum - array_extent.minimum);

        Self {
            array_shape: array_extent.shape,
            index_min,
            iter_extent,
        }
    }

    #[inline]
    pub fn new_global(array_extent: &ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(array_extent);

        Self::new_global_unchecked(array_extent, iter_extent)
    }
}

impl<N> ArrayExtentVisitor<N>
where
    N: ArrayIndexer<N>,
    PointN<N>: Copy,
{
    pub fn for_each_point_and_stride(&self, f: impl FnMut(PointN<N>, Stride)) {
        N::for_each_point_and_stride_unchecked(
            self.array_shape,
            self.index_min,
            self.iter_extent,
            f,
        )
    }
}
