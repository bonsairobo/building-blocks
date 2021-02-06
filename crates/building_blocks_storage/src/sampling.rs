use crate::{prelude::*, ArrayIndexer};

use building_blocks_core::prelude::*;

pub trait ChunkDownsampler<N, T> {
    /// Samples `src_chunk` in order to write out just a portion of `dst_chunk`, starting at `dst_min`.
    fn downsample(
        &self,
        src_chunk: &ArrayN<N, T>,
        dst_chunk: &mut ArrayN<N, T>,
        dst_min: Local<N>,
        level_delta: u8,
    );
}

pub struct PointDownsampler;

impl<N, T> ChunkDownsampler<N, T> for PointDownsampler
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    T: Copy,
{
    fn downsample(
        &self,
        src_chunk: &ArrayN<N, T>,
        dst_chunk: &mut ArrayN<N, T>,
        dst_min: Local<N>,
        lod_delta: u8,
    ) {
        // PERF: this might be faster using Strides

        debug_assert!(lod_delta > 0);
        let lod_delta = lod_delta as i32;

        let sample_shape = src_chunk.extent().shape >> lod_delta;
        debug_assert!(sample_shape > PointN::ZERO);

        for p in ExtentN::from_min_and_shape(PointN::ZERO, sample_shape).iter_points() {
            *dst_chunk.get_mut(Local(dst_min.0 + p)) = src_chunk.get(Local(p << lod_delta));
        }
    }
}
