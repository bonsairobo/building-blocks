mod point;
mod sdf_mean;

pub use point::*;
pub use sdf_mean::*;

use super::child_mask_has_child;
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

impl<N, T, Usr, Bldr, Store> ChunkTree<N, T, Bldr, Store>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: FillExtent<N, Item = T> + IndexedArray<N>,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr> + for<'r> IterChunkKeys<'r, N>,
{
    /// Downsamples all chunks in level `src_lod` that overlap `src_extent`.
    ///
    /// All ancestors from `src_lod + 1` to `max_lod` (inclusive) will be targeted as sample destinations. Levels from `max_lod
    /// + 1` to the roots will not be touched.
    pub fn downsample_extent<Samp>(
        &mut self,
        sampler: &Samp,
        src_lod: u8,
        max_lod: u8,
        src_extent: ExtentN<N>,
    ) where
        Samp: ChunkDownsampler<N, T, Usr, Usr>,
    {
        self.downsample_extent_internal(
            |this, src_chunk_key| this.downsample_chunk(sampler, src_chunk_key),
            src_lod,
            max_lod,
            src_extent,
        );
    }

    /// Same as `downsample_extent`, but allows passing in a closure that fetches LOD0 chunks. This is mostly a workaround so we
    /// can downsample multichannel chunks from LOD0.
    pub fn downsample_extent_with_lod0<Samp, Lod0Ch, Lod0ChBorrow>(
        &mut self,
        get_lod0_chunk: impl Fn(PointN<N>) -> Option<Lod0Ch>,
        sampler: &Samp,
        src_lod: u8,
        max_lod: u8,
        src_extent: ExtentN<N>,
    ) where
        Lod0Ch: Borrow<Lod0ChBorrow>,
        Samp: ChunkDownsampler<N, T, Usr, Usr> + ChunkDownsampler<N, T, Lod0ChBorrow, Usr>,
    {
        self.downsample_extent_internal(
            |this, src_chunk_key| {
                if src_chunk_key.lod == 0 {
                    if let Some(src_chunk) = get_lod0_chunk(src_chunk_key.minimum) {
                        this.downsample_external_chunk(sampler, src_chunk_key, src_chunk.borrow());
                    } else {
                        this.downsample_ambient_chunk(src_chunk_key);
                    }
                } else {
                    this.downsample_chunk(sampler, src_chunk_key);
                }
            },
            src_lod,
            max_lod,
            src_extent,
        );
    }

    fn downsample_extent_internal(
        &mut self,
        mut downsample_fn: impl FnMut(&mut Self, ChunkKey<N>),
        src_lod: u8,
        max_lod: u8,
        src_extent: ExtentN<N>,
    ) {
        let root_lod = self.root_lod();
        assert!(src_lod < root_lod);
        assert!(max_lod <= root_lod);
        let root_keys: Vec<_> = self.lod_storage(root_lod).chunk_keys().cloned().collect();
        for root_chunk_min in root_keys.into_iter() {
            self.downsample_extent_internal_recursive(
                &mut downsample_fn,
                ChunkKey::new(root_lod, root_chunk_min),
                src_lod,
                max_lod,
                &src_extent,
            );
        }
    }

    fn downsample_extent_internal_recursive(
        &mut self,
        downsample_fn: &mut impl FnMut(&mut Self, ChunkKey<N>),
        node_key: ChunkKey<N>,
        src_lod: u8,
        max_lod: u8,
        src_extent: &ExtentN<N>,
    ) {
        if node_key.lod > src_lod {
            if let Some(node) = self.get_node(node_key) {
                let child_mask = node.child_mask;
                for child_i in 0..PointN::NUM_CORNERS {
                    if child_mask_has_child(child_mask, child_i) {
                        let child_key = self.indexer.child_chunk_key(node_key, child_i);

                        // Only visit chunks overlapping src_extent and ancestors.
                        if self
                            .indexer
                            .chunk_extent_at_lower_lod(child_key, src_lod)
                            .intersection(src_extent)
                            .is_empty()
                        {
                            continue;
                        }

                        self.downsample_extent_internal_recursive(
                            downsample_fn,
                            child_key,
                            src_lod,
                            max_lod,
                            src_extent,
                        );
                    }
                }
            }
        }

        // Do a post-order traversal so we can start sampling from the bottom of the tree and work our way up.
        if max_lod > node_key.lod && node_key.lod >= src_lod {
            downsample_fn(self, node_key);
        }
    }

    /// Downsamples the chunk at `src_chunk_key` into `lod + 1`.
    fn downsample_chunk<Samp>(&mut self, sampler: &Samp, src_chunk_key: ChunkKey<N>)
    where
        Samp: ChunkDownsampler<N, T, Usr, Usr>,
    {
        // PERF: Unforunately we have to remove the chunk and put it back to satisfy the borrow checker.
        if let Some(src_node) = self.pop_node(src_chunk_key) {
            if let Some(src_chunk) = &src_node.user_chunk {
                self.downsample_external_chunk(sampler, src_chunk_key, src_chunk);
            } else {
                self.downsample_ambient_chunk(src_chunk_key);
            }
            self.write_node(src_chunk_key, src_node);
        } else {
            self.downsample_ambient_chunk(src_chunk_key)
        }
    }

    /// Downsamples `src_chunk` into `src_chunk_key.lod + 1`.
    fn downsample_external_chunk<Samp, Src>(
        &mut self,
        sampler: &Samp,
        src_chunk_key: ChunkKey<N>,
        src_chunk: &Src,
    ) where
        Samp: ChunkDownsampler<N, T, Src, Usr>,
    {
        let dst = self.indexer.downsample_destination(src_chunk_key);
        let dst_chunk = self.get_mut_chunk_or_insert_ambient(dst.chunk_key);
        sampler.downsample(src_chunk, dst_chunk, dst.offset);
    }

    /// Downsamples an ambient chunk into `lod + 1`. This simply fills the destination extent with the ambient value.
    fn downsample_ambient_chunk(&mut self, src_chunk_key: ChunkKey<N>) {
        let chunk_shape = self.chunk_shape();
        let ambient_value = self.ambient_value.clone();
        let dst = self.indexer.downsample_destination(src_chunk_key);
        let dst_chunk = self.get_mut_chunk_or_insert_ambient(dst.chunk_key);
        let dst_array = dst_chunk.array_mut();
        let dst_extent = ExtentN::from_min_and_shape(
            dst_array.extent().minimum + dst.offset.0,
            chunk_shape >> 1,
        );
        dst_array.fill_extent(&dst_extent, ambient_value);
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
    fn downsample_multichannel_chunks() {
        let chunk_shape = Point3i::fill(16);
        let ambient = (Sd8::ONE, 'a');

        let lod0_extent =
            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(2)) * chunk_shape;

        // Build a multichannel chunk map for LOD0.
        let lod0_builder = ChunkTreeBuilder3x2::new(ChunkTreeConfig {
            chunk_shape,
            ambient_value: ambient,
            root_lod: 0,
        });
        let mut lod0 = lod0_builder.build_with_hash_map_storage();
        lod0.fill_extent(0, &lod0_extent, ambient);

        // Build a single-channel chunk map for LOD > 0.
        let lodn_builder = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
            chunk_shape,
            ambient_value: Sd8::ONE,
            root_lod: 5,
        });
        let mut lodn = lodn_builder.build_with_hash_map_storage();

        // Since we're downsampling multichannel chunks, we need to project them onto the one channel that we're downsampling.
        let get_lod0_chunk = |p| {
            lod0.get_chunk(ChunkKey::new(0, p))
                .map(|chunk| chunk.borrow_channels(|(sd, _letter)| sd))
        };

        // Only keep samples at 4 levels.
        lodn.downsample_extent_with_lod0(get_lod0_chunk, &SdfMeanDownsampler, 0, 3, lod0_extent);
    }
}
