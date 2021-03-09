use crate::{
    for_each_stride_parallel_global_unchecked2, for_each_stride_parallel_global_unchecked3,
    Array2ForEach, Array3ForEach, ArrayForEach, Local, Local2i, Local3i, Stride,
};

use building_blocks_core::prelude::*;

/// When a lattice map implements `Array`, that means there is some underlying array with the location and shape dictated by the
/// extent.
///
/// For the sake of generic impls, if the same map also implements `Get*<Stride>`, it must use the same data layout as `ArrayN`.
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

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each_point_and_stride_unchecked(
        for_each: ArrayForEach<N>,
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

impl ArrayIndexer<[i32; 2]> for [i32; 2] {
    #[inline]
    fn stride_from_local_point(s: Point2i, p: Local2i) -> Stride {
        Stride((p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each_point_and_stride_unchecked(
        for_each: Array2ForEach,
        mut f: impl FnMut(Point2i, Stride),
    ) {
        for_each2!(for_each, x, y, stride, { f(PointN([x, y]), stride) });
    }

    #[inline]
    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &Extent2i,
        array1_extent: &Extent2i,
        array2_extent: &Extent2i,
        f: impl FnMut(Stride, Stride),
    ) {
        for_each_stride_parallel_global_unchecked2(iter_extent, array1_extent, array2_extent, f)
    }
}

impl ArrayIndexer<[i32; 3]> for [i32; 3] {
    #[inline]
    fn stride_from_local_point(s: Point3i, p: Local3i) -> Stride {
        Stride((p.z() * s.y() * s.x() + p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each_point_and_stride_unchecked(
        for_each: Array3ForEach,
        mut f: impl FnMut(Point3i, Stride),
    ) {
        for_each3!(for_each, x, y, z, stride, { f(PointN([x, y, z]), stride) });
    }

    #[inline]
    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &Extent3i,
        array1_extent: &Extent3i,
        array2_extent: &Extent3i,
        f: impl FnMut(Stride, Stride),
    ) {
        for_each_stride_parallel_global_unchecked3(iter_extent, array1_extent, array2_extent, f);
    }
}
