use crate::{
    for_each2, for_each3, Array2ForEach, Array3ForEach, ArrayForEach, ArrayStrideIter, Local,
    Local2i, Local3i, LockStepArrayForEach, LockStepArrayForEach2, LockStepArrayForEach3, Stride,
};

use building_blocks_core::prelude::*;

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn make_stride_iter(
        array_shape: PointN<N>,
        origin: Local<N>,
        step: PointN<N>,
    ) -> ArrayStrideIter;

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
    fn make_stride_iter(array_shape: Point2i, origin: Local2i, step: Point2i) -> ArrayStrideIter {
        ArrayStrideIter::new_2d(array_shape, origin, step)
    }

    #[inline]
    fn for_each(for_each: Array2ForEach, f: impl FnMut(Point2i, Stride)) {
        let Array2ForEach { iter_extent, iter } = for_each;
        for_each2(iter, &iter_extent, f);
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        for_each: LockStepArrayForEach2,
        f: impl FnMut(Point2i, (Stride, Stride)),
    ) {
        let LockStepArrayForEach2 {
            iter_extent,
            iter1,
            iter2,
        } = for_each;
        for_each2((iter1, iter2), &iter_extent, f);
    }
}

impl ArrayIndexer<[i32; 3]> for [i32; 3] {
    #[inline]
    fn stride_from_local_point(s: Point3i, p: Local3i) -> Stride {
        Stride((p.z() * s.y() * s.x() + p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn make_stride_iter(array_shape: Point3i, origin: Local3i, step: Point3i) -> ArrayStrideIter {
        ArrayStrideIter::new_3d(array_shape, origin, step)
    }

    #[inline]
    fn for_each(for_each: Array3ForEach, f: impl FnMut(Point3i, Stride)) {
        let Array3ForEach { iter_extent, iter } = for_each;
        for_each3(iter, &iter_extent, f);
    }

    #[inline]
    fn for_each_lockstep_unchecked(
        for_each: LockStepArrayForEach3,
        f: impl FnMut(Point3i, (Stride, Stride)),
    ) {
        let LockStepArrayForEach3 {
            iter_extent,
            iter1,
            iter2,
        } = for_each;
        for_each3((iter1, iter2), &iter_extent, f)
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
