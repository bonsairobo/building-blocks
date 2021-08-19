use super::chunk_downsample_for_each;
use crate::{
    array::ArrayIndexer,
    dev_prelude::{ChunkDownsampler, GetMutUnchecked, GetUnchecked, IndexedArray, Local, Stride},
};

use building_blocks_core::prelude::*;

/// A `ChunkDownsampler` that just selects a single point from each `2x2x2` region, ignoring the rest.
pub struct PointDownsampler;

impl<N, Src, Dst, T> ChunkDownsampler<N, Src, Dst> for PointDownsampler
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
    Src: GetUnchecked<Stride, Item = T> + IndexedArray<N>,
    Dst: for<'r> GetMutUnchecked<'r, Stride, Item = &'r mut T>,
{
    fn downsample(&self, src_chunk: &Src, dst_chunk: &mut Dst, dst_min: Local<N>) {
        let chunk_shape = src_chunk.extent().shape; // Doesn't matter which chunk we choose, they should have the same shape.
        let for_each = chunk_downsample_for_each(chunk_shape, dst_min);
        N::for_each_lockstep_unchecked(for_each, |_p, (s_dst, s_src)| unsafe {
            *dst_chunk.get_mut_unchecked(s_dst) = src_chunk.get_unchecked(s_src);
        });
    }
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
    use crate::prelude::{Array3x1, ForEach};

    #[test]
    fn point_downsample_only_ones() {
        let step = 2;

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
        PointDownsampler.downsample(&src_chunk, &mut dst_chunk, dst_min);

        let dst_extent = chunk_extent >> 1;
        dst_chunk.for_each(&dst_extent, |p: Point3i, x| assert_eq!(x, 1, "p = {:?}", p));
    }
}
