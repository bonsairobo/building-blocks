use crate::{
    for_each2, for_each3, Array2ForEach, Array2StrideIter, Array3ForEach, Array3StrideIter,
    ArrayForEach, Local, Local2i, Local3i, Stride,
};

use building_blocks_core::prelude::*;

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each(for_each: ArrayForEach<N>, f: impl FnMut(PointN<N>, Stride));

    fn for_each_lockstep_unchecked(
        iter_extent: &ExtentN<N>,
        array1_extent: &ExtentN<N>,
        array2_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, (Stride, Stride)),
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

impl ArrayIndexer<[i32; 2]> for [i32; 2] {
    #[inline]
    fn stride_from_local_point(s: Point2i, p: Local2i) -> Stride {
        Stride((p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each(for_each: Array2ForEach, f: impl FnMut(Point2i, Stride)) {
        let Array2ForEach {
            iter_extent,
            array_shape,
            span,
        } = for_each;
        for_each2(
            Array2StrideIter::new_with_step(array_shape, span.origin, span.step),
            &iter_extent,
            f,
        );
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        iter_extent: &Extent2i,
        array1_extent: &Extent2i,
        array2_extent: &Extent2i,
        f: impl FnMut(Point2i, (Stride, Stride)),
    ) {
        // Translate to local coordinates.
        let min1 = iter_extent.minimum - array1_extent.minimum;
        let min2 = iter_extent.minimum - array2_extent.minimum;

        let s1 = Array2StrideIter::new(array1_extent.shape, Local(min1));
        let s2 = Array2StrideIter::new(array2_extent.shape, Local(min2));

        for_each2((s1, s2), iter_extent, f);
    }
}

impl ArrayIndexer<[i32; 3]> for [i32; 3] {
    #[inline]
    fn stride_from_local_point(s: Point3i, p: Local3i) -> Stride {
        Stride((p.z() * s.y() * s.x() + p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each(for_each: Array3ForEach, f: impl FnMut(Point3i, Stride)) {
        let Array3ForEach {
            iter_extent,
            array_shape,
            span,
        } = for_each;
        for_each3(
            Array3StrideIter::new_with_step(array_shape, span.origin, span.step),
            &iter_extent,
            f,
        );
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        iter_extent: &Extent3i,
        array1_extent: &Extent3i,
        array2_extent: &Extent3i,
        f: impl FnMut(Point3i, (Stride, Stride)),
    ) {
        // Translate to local coordinates.
        let min1 = iter_extent.minimum - array1_extent.minimum;
        let min2 = iter_extent.minimum - array2_extent.minimum;

        let s1 = Array3StrideIter::new(array1_extent.shape, Local(min1));
        let s2 = Array3StrideIter::new(array2_extent.shape, Local(min2));

        for_each3((s1, s2), iter_extent, f)
    }
}

/// When a lattice map implements `IndexedArray`, that means there is some underlying array with the location and shape dictated
/// by the extent.
///
/// For the sake of generic impls, if the same map also implements `Get*<Stride>`, it must use the same data layout as `Array`.
pub trait IndexedArray<N> {
    type Indexer: ArrayIndexer<N>;

    fn extent(&self) -> &ExtentN<N>;

    #[inline]
    fn stride_from_local_point(&self, p: Local<N>) -> Stride
    where
        PointN<N>: Copy,
    {
        Self::Indexer::stride_from_local_point(self.extent().shape, p)
    }

    #[inline]
    fn strides_from_local_points(&self, points: &[Local<N>], strides: &mut [Stride])
    where
        PointN<N>: Copy,
    {
        Self::Indexer::strides_from_local_points(self.extent().shape, points, strides)
    }
}
