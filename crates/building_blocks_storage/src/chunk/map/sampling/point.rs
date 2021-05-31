use crate::{ArrayIndexer, ArrayNx1, ChunkDownsampler, Get, GetMut, IndexedArray, Local};

use building_blocks_core::prelude::*;

/// A `ChunkDownsampler` that just selects a single point from each `2x2x2` region, discarding the rest.
pub struct PointDownsampler;

impl<N, Src, T> ChunkDownsampler<N, T, Src> for PointDownsampler
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    T: 'static + Copy,
    Src: Get<Local<N>, Item = T> + IndexedArray<N>,
{
    fn downsample(
        &self,
        src_chunk: &Src,
        dst_chunk: &mut ArrayNx1<N, T>,
        dst_min: Local<N>,
        lod_delta: u8,
    ) {
        // PERF: this might be faster using Strides

        debug_assert!(lod_delta > 0);
        let lod_delta = lod_delta as i32;

        let dst_shape = src_chunk.extent().shape >> lod_delta;
        debug_assert!(dst_shape > PointN::ZERO);

        for p in ExtentN::from_min_and_shape(PointN::ZERO, dst_shape).iter_points() {
            *dst_chunk.get_mut(Local(dst_min.0 + p)) = src_chunk.get(Local(p << lod_delta));
        }
    }
}
