pub mod point;
pub mod sdf_mean;

pub use point::*;
pub use sdf_mean::*;

use crate::{
    array::{ArrayIndexer, LockStepArrayForEach},
    dev_prelude::*,
};

use building_blocks_core::prelude::*;
use std::borrow::Borrow;

pub trait ChunkDownsampler<N, T, Src, Dst> {
    /// Samples `src_chunk` in order to write out just a portion of `dst_chunk`, starting at `dst_min`, where the destination
    /// has half the resolution (sample rate) of the source.
    fn downsample(&self, src_chunk: &Src, dst_chunk: &mut Dst, dst_min: Local<N>);
}

impl<N, T, Ch, Bldr, Store> ChunkMap<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    ChunkKey<N>: Copy,
    T: Clone,
    Ch: Chunk + FillExtent<N, Item = T> + IndexedArray<N>,
    Bldr: ChunkMapBuilder<N, T, Chunk = Ch>,
    Store: ChunkStorage<N, Chunk = Ch>,
{
    /// Downsamples the chunk at `src_chunk_key` into the specified destination level `dst_lod`.
    pub fn downsample_chunk<Samp>(
        &mut self,
        sampler: &Samp,
        src_chunk_key: ChunkKey<N>,
        dst_lod: u8,
    ) where
        Samp: ChunkDownsampler<N, T, Ch, Ch>,
    {
        // PERF: Unforunately we have to remove the chunk and put it back to satisfy the borrow checker.
        if let Some(src_chunk) = self.pop_chunk(src_chunk_key) {
            self.downsample_external_chunk(sampler, src_chunk_key, &src_chunk, dst_lod);
            self.write_chunk(src_chunk_key, src_chunk);
        } else {
            self.downsample_ambient_chunk(src_chunk_key, dst_lod)
        }
    }

    /// Downsamples all of `src_chunk` into the overlapping chunk at level `dst_lod`.
    pub fn downsample_external_chunk<Samp, Src>(
        &mut self,
        sampler: &Samp,
        src_chunk_key: ChunkKey<N>,
        src_chunk: &Src,
        dst_lod: u8,
    ) where
        Samp: ChunkDownsampler<N, T, Src, Ch>,
    {
        assert!(dst_lod > src_chunk_key.lod);

        let dst = self.indexer.downsample_destination(src_chunk_key.minimum);
        let dst_chunk =
            self.get_mut_chunk_or_insert_ambient(ChunkKey::new(dst_lod, dst.dst_chunk_min));
        sampler.downsample(src_chunk, dst_chunk, dst.dst_offset);
    }

    /// Fill the destination samples with the ambient value.
    pub fn downsample_ambient_chunk(&mut self, src_chunk_key: ChunkKey<N>, dst_lod: u8) {
        assert!(dst_lod > src_chunk_key.lod);

        let chunk_shape = self.chunk_shape();
        let ambient_value = self.ambient_value.clone();
        let dst = self.indexer.downsample_destination(src_chunk_key.minimum);
        let dst_chunk =
            self.get_mut_chunk_or_insert_ambient(ChunkKey::new(dst_lod, dst.dst_chunk_min));
        let dst_extent = ExtentN::from_min_and_shape(
            dst_chunk.extent().minimum + dst.dst_offset.0,
            chunk_shape >> 1,
        );
        dst_chunk.fill_extent(&dst_extent, ambient_value);
    }
}

impl<T, Ch, Bldr, Store> ChunkMap3<T, Bldr, Store>
where
    T: Clone,
    Ch: Chunk + FillExtent<[i32; 3], Item = T> + IndexedArray<[i32; 3]>,
    Bldr: ChunkMapBuilder<[i32; 3], T, Chunk = Ch>,
    Store: ChunkStorage<[i32; 3], Chunk = Ch>,
{
    /// Downsamples all chunks that both:
    ///   1. overlap `extent`
    ///   2. are present in `index`
    ///
    /// Destination chunks up to `num_lods` will be considered.
    pub fn downsample_chunks_with_index<Samp>(
        &mut self,
        index: &OctreeChunkIndex,
        sampler: &Samp,
        extent: &Extent3i,
    ) where
        Samp: ChunkDownsampler<[i32; 3], T, Ch, Ch>,
    {
        let chunk_shape = self.chunk_shape();
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);

        let chunk_space_extent = *extent >> chunk_log2;

        index.visit_octrees(extent, &mut |octree| {
            // Post-order is important to make sure we start downsampling at LOD 0.
            octree.visit_all_octants_for_extent_in_postorder(
                &chunk_space_extent,
                &mut |node: &OctreeNode| {
                    let src_lod = node.octant().exponent();
                    let dst_lod = src_lod + 1;
                    if dst_lod < index.num_lods() {
                        let src_chunk_min = (node.octant().minimum() << chunk_log2) >> src_lod;
                        self.downsample_chunk(
                            sampler,
                            ChunkKey::new(src_lod, src_chunk_min),
                            dst_lod,
                        );
                    }

                    VisitStatus::Continue
                },
            );
        });
    }

    /// Same as `downsample_chunks_with_index`, but allows passing in a closure that fetches LOD0 chunks. This is mostly a
    /// workaround so we can downsample multichannel chunks from LOD0.
    pub fn downsample_chunks_with_lod0_and_index<Samp, Lod0Ch, Lod0ChBorrow>(
        &mut self,
        get_lod0_chunk: impl Fn(Point3i) -> Option<Lod0Ch>,
        index: &OctreeChunkIndex,
        sampler: &Samp,
        extent: &Extent3i,
    ) where
        Lod0Ch: Borrow<Lod0ChBorrow>,
        Samp:
            ChunkDownsampler<[i32; 3], T, Ch, Ch> + ChunkDownsampler<[i32; 3], T, Lod0ChBorrow, Ch>,
    {
        let chunk_shape = self.chunk_shape();
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);

        let chunk_space_extent = *extent >> chunk_log2;

        index.visit_octrees(extent, &mut |octree| {
            // Post-order is important to make sure we start downsampling at LOD 0.
            octree.visit_all_octants_for_extent_in_postorder(
                &chunk_space_extent,
                &mut |node: &OctreeNode| {
                    let src_lod = node.octant().exponent();
                    let dst_lod = src_lod + 1;
                    if dst_lod < index.num_lods() {
                        let src_chunk_min = (node.octant().minimum() << chunk_log2) >> src_lod;
                        let src_chunk_key = ChunkKey::new(src_lod, src_chunk_min);

                        if src_lod == 0 {
                            if let Some(src_chunk) = get_lod0_chunk(src_chunk_min) {
                                self.downsample_external_chunk(
                                    sampler,
                                    src_chunk_key,
                                    src_chunk.borrow(),
                                    dst_lod,
                                );
                            } else {
                                self.downsample_ambient_chunk(src_chunk_key, dst_lod);
                            }
                        } else {
                            self.downsample_chunk(sampler, src_chunk_key, dst_lod);
                        }
                    }

                    VisitStatus::Continue
                },
            );
        });
    }
}

fn chunk_downsample_for_each<N>(
    chunk_shape: PointN<N>,
    dst_min: Local<N>,
) -> LockStepArrayForEach<N>
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint<N>,
{
    let dst_shape = chunk_shape >> 1;
    debug_assert!(dst_shape > PointN::ZERO);

    let iter_extent = ExtentN::from_min_and_shape(PointN::ZERO, dst_shape);
    let dst_iter = N::make_stride_iter(chunk_shape, dst_min, PointN::ONES);
    let src_iter = N::make_stride_iter(chunk_shape, Local(PointN::ZERO), PointN::fill(2));

    LockStepArrayForEach::new(iter_extent, dst_iter, src_iter)
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{Sd8, SdfMeanDownsampler};

    #[test]
    fn downsample_multichannel_chunks_with_index() {
        let num_lods = 6;
        let chunk_shape = Point3i::fill(16);

        let lod0_extent =
            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(2)) * chunk_shape;

        // Build a multichannel chunk map for LOD0.
        let ambient = (Sd8::ONE, 'a');
        let lod0_builder = ChunkMapBuilder3x2::new(chunk_shape, ambient);
        let mut lod0 = lod0_builder.build_with_hash_map_storage();
        lod0.fill_extent(0, &lod0_extent, ambient);

        let lodn_builder = ChunkMapBuilder3x1::new(chunk_shape, Sd8::ONE);
        let mut lodn = lodn_builder.build_with_hash_map_storage();

        let index = OctreeChunkIndex::index_chunk_map(9, num_lods, &lod0);

        // Since we're downsampling multichannel chunks, we need to project them onto the one channel that we're downsampling.
        let get_lod0_chunk = |p| {
            lod0.get_chunk(ChunkKey::new(0, p))
                .map(|chunk| chunk.borrow_channels(|(sd, _letter)| sd))
        };

        lodn.downsample_chunks_with_lod0_and_index(
            get_lod0_chunk,
            &index,
            &SdfMeanDownsampler,
            &lod0_extent,
        );
    }
}
