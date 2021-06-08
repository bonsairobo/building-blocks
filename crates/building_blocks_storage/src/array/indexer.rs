use crate::{
    for_each2, for_each3, Array2ForEach, Array2StrideIter, Array3ForEach, Array3StrideIter,
    ArrayForEach, ArrayIterSpan, Local, Local2i, Local3i, LockStepArrayForEach,
    LockStepArrayForEach2, LockStepArrayForEach3, Stride,
};

use building_blocks_core::prelude::*;

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each(for_each: ArrayForEach<N>, f: impl FnMut(PointN<N>, Stride));

    fn for_each_lockstep_unchecked(
        for_each: LockStepArrayForEach<N>,
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
            span:
                ArrayIterSpan {
                    array_shape,
                    origin,
                    step,
                },
        } = for_each;
        for_each2(
            Array2StrideIter::new(array_shape, origin, step),
            &iter_extent,
            f,
        );
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        for_each: LockStepArrayForEach2,
        f: impl FnMut(Point2i, (Stride, Stride)),
    ) {
        let LockStepArrayForEach2 {
            iter_extent,
            span1,
            span2,
        } = for_each;
        let s1 = Array2StrideIter::new(span1.array_shape, span1.origin, span1.step);
        let s2 = Array2StrideIter::new(span2.array_shape, span2.origin, span2.step);

        for_each2((s1, s2), &iter_extent, f);
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
            span:
                ArrayIterSpan {
                    array_shape,
                    origin,
                    step,
                },
        } = for_each;
        for_each3(
            Array3StrideIter::new(array_shape, origin, step),
            &iter_extent,
            f,
        );
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        for_each: LockStepArrayForEach3,
        f: impl FnMut(Point3i, (Stride, Stride)),
    ) {
        let LockStepArrayForEach3 {
            iter_extent,
            span1,
            span2,
        } = for_each;
        let s1 = Array3StrideIter::new(span1.array_shape, span1.origin, span1.step);
        let s2 = Array3StrideIter::new(span2.array_shape, span2.origin, span2.step);

        for_each3((s1, s2), &iter_extent, f)
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
