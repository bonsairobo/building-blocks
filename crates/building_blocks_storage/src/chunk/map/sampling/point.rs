use crate::{
    ArrayIndexer, ArrayIterSpan, ChunkDownsampler, Get, GetMut, IndexedArray, Local,
    LockStepArrayForEach, Stride,
};

use building_blocks_core::prelude::*;

/// A `ChunkDownsampler` that just selects a single point from each `2x2x2` region, discarding the rest.
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
