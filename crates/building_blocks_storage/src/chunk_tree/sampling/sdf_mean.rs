use super::chunk_downsample_for_each;
use crate::{
    array::{ArrayForEach, ArrayIndexer},
    dev_prelude::{ChunkDownsampler, IndexedArray, Local, Stride},
    prelude::{GetMutUnchecked, GetUnchecked},
};

use building_blocks_core::prelude::*;

/// A `ChunkDownsampler` that takes the mean of each `2x2x2` region of a signed distance field. It also renormalizes the values
/// to lie in the range `[-1.0, 1.0]`.
pub struct SdfMeanDownsampler;

impl<N, Src, Dst, T> ChunkDownsampler<N, Src, Dst> for SdfMeanDownsampler
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint,
    ArrayForEach<N>: Clone,
    T: From<f32>,
    f32: From<T>,
    Src: GetUnchecked<Stride, Item = T> + IndexedArray<N>,
    Dst: for<'r> GetMutUnchecked<'r, Stride, Item = &'r mut T> + IndexedArray<N>,
{
    fn downsample(&self, src_chunk: &Src, dst_chunk: &mut Dst, dst_min: Local<N>) {
        let chunk_shape = src_chunk.extent().shape; // Doesn't matter which chunk we choose, they should have the same shape.

        let src_shape_per_point = PointN::fill(2);

        let kernel_for_each = ArrayForEach::new_local_unchecked(
            chunk_shape,
            Local(PointN::ZERO),
            ExtentN::from_min_and_shape(PointN::ZERO, src_shape_per_point),
        );

        // Not only do we get the mean signed distance value by dividing by the volume, but we also re-normalize by dividing
        // by the scale factor (the ratio between voxel edge lengths at the different resolutions).
        let rescale = 1.0 / (2 * src_shape_per_point.volume()) as f32;

        let for_each = chunk_downsample_for_each(chunk_shape, dst_min);
        N::for_each_lockstep_unchecked(for_each, |_p, (s_dst, s_src)| {
            let mut sum = 0.0;
            N::for_each(kernel_for_each.clone(), |_p, neighbor_offset| {
                sum += f32::from(unsafe { src_chunk.get_unchecked(s_src + neighbor_offset) });
            });
            unsafe {
                *dst_chunk.get_mut_unchecked(s_dst) = T::from(rescale * sum);
            }
        });
    }
}
