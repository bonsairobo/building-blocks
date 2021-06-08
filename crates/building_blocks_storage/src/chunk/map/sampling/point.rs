use crate::{
    ArrayIndexer, ArrayIterSpan, ChunkDownsampler, Get, GetMut, IndexedArray, Local,
    LockStepArrayForEach, Stride,
};

use building_blocks_core::prelude::*;

/// A `ChunkDownsampler` that just selects a single point from each `2x2x2` (assuming `lod_delta=1`) region, ignoring the rest.
pub struct PointDownsampler;

impl<N, Src, Dst, T> ChunkDownsampler<N, T, Src, Dst> for PointDownsampler
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    Src: Get<Stride, Item = T> + IndexedArray<N>,
    Dst: for<'r> GetMut<'r, Stride, Item = &'r mut T>,
{
    fn downsample(&self, src_chunk: &Src, dst_chunk: &mut Dst, dst_min: Local<N>, lod_delta: u8) {
        debug_assert!(lod_delta > 0);
        let lod_delta = lod_delta as i32;
        let chunk_shape = src_chunk.extent().shape; // Doesn't matter which chunk we choose, they should have the same shape.
        let for_each = lock_step_for_each(chunk_shape, dst_min, lod_delta);
        N::for_each_lockstep_unchecked(for_each, |_p, (s_dst, s_src)| {
            *dst_chunk.get_mut(s_dst) = src_chunk.get(s_src);
        });
    }
}

fn lock_step_for_each<N>(
    chunk_shape: PointN<N>,
    dst_min: Local<N>,
    lod_delta: i32,
) -> LockStepArrayForEach<N>
where
    PointN<N>: IntegerPoint<N>,
{
    let dst_shape = chunk_shape >> lod_delta;
    debug_assert!(dst_shape > PointN::ZERO);

    let iter_extent = ExtentN::from_min_and_shape(PointN::ZERO, dst_shape);
    let span_dst = ArrayIterSpan {
        array_shape: chunk_shape,
        origin: dst_min,
        step: PointN::ONES,
    };
    let span_src = ArrayIterSpan {
        array_shape: chunk_shape,
        origin: Local(PointN::ZERO),
        step: PointN::ONES << lod_delta,
    };

    LockStepArrayForEach::new(iter_extent, span_dst, span_src)
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Array3x1, ForEach};

    #[test]
    fn point_downsample_only_ones() {
        let lod_delta = 1;
        let step = 1 << lod_delta;

        let chunk_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));

        // Make an array where only points with components divisible by 2 have value 1. These are exactly the points that will
        // be sampled.
        let src_chunk = Array3x1::fill_with(chunk_extent, |p| {
            if p.x() % step == 0 && p.y() % step == 0 && p.z() % step == 0 {
                1
            } else {
                0
            }
        });

        let mut dst_chunk = Array3x1::fill(chunk_extent, 0);
        let dst_min = Local(Point3i::ZERO);
        PointDownsampler.downsample(&src_chunk, &mut dst_chunk, dst_min, lod_delta);

        let dst_extent =
            Extent3i::from_min_and_shape(Point3i::ZERO, chunk_extent.shape >> lod_delta as i32);
        dst_chunk.for_each(&dst_extent, |p: Point3i, x| assert_eq!(x, 1, "p = {:?}", p));
    }
}
