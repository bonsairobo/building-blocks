mod point;
mod sdf_mean;

pub use point::*;
pub use sdf_mean::*;

use crate::{
    array::{ArrayIndexer, LockStepArrayForEach},
    dev_prelude::*,
};

use building_blocks_core::prelude::*;
use std::borrow::Borrow;

pub trait ChunkDownsampler<N, Src, Dst> {
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
    pub fn downsample_extent_into_self<Samp>(
        &mut self,
        sampler: &Samp,
        src_lod: u8,
        max_lod: u8,
        src_extent: ExtentN<N>,
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, Usr::Array>,
    {
        self.downsample_extent_into_self_internal(
            |this, src_chunk_key| this.downsample_into_self(sampler, src_chunk_key),
            src_lod,
            max_lod,
            src_extent,
        );
    }

    /// Same as `downsample_extent`, but allows passing in a closure that fetches LOD0 chunks. This is mostly a workaround so we
    /// can downsample multichannel chunks from LOD0.
    pub fn downsample_extent_into_self_with_lod0<Samp, Lod0Ch, Lod0ChBorrow>(
        &mut self,
        get_lod0_chunk: impl Fn(PointN<N>) -> Option<Lod0Ch>,
        sampler: &Samp,
        src_lod: u8,
        max_lod: u8,
        src_extent: ExtentN<N>,
    ) where
        Lod0Ch: Borrow<Lod0ChBorrow>,
        Lod0ChBorrow: UserChunk,
        Samp: ChunkDownsampler<N, Usr::Array, Usr::Array>
            + ChunkDownsampler<N, Lod0ChBorrow::Array, Usr::Array>,
    {
        self.downsample_extent_into_self_internal(
            |this, src_chunk_key| {
                if src_chunk_key.lod == 0 {
                    if let Some(src_chunk) = get_lod0_chunk(src_chunk_key.minimum) {
                        this.downsample_external_into_self(
                            sampler,
                            src_chunk_key,
                            src_chunk.borrow(),
                        );
                    } else {
                        this.downsample_ambient_into_self(src_chunk_key);
                    }
                } else {
                    this.downsample_into_self(sampler, src_chunk_key);
                }
            },
            src_lod,
            max_lod,
            src_extent,
        );
    }

    fn downsample_extent_into_self_internal(
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
            self.downsample_extent_into_self_internal_recursive(
                &mut downsample_fn,
                ChunkKey::new(root_lod, root_chunk_min),
                src_lod,
                max_lod,
                &src_extent,
            );
        }
    }

    fn downsample_extent_into_self_internal_recursive(
        &mut self,
        downsample_fn: &mut impl FnMut(&mut Self, ChunkKey<N>),
        node_key: ChunkKey<N>,
        src_lod: u8,
        max_lod: u8,
        src_extent: &ExtentN<N>,
    ) {
        if node_key.lod > src_lod {
            if let Some((state, _)) = self.get_node_state(node_key) {
                let state = state.clone();
                for child_i in 0..PointN::NUM_CORNERS {
                    if state.children.bit_is_set(child_i) {
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

                        self.downsample_extent_into_self_internal_recursive(
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
    pub fn downsample_into_self<Samp>(&mut self, sampler: &Samp, src_chunk_key: ChunkKey<N>)
    where
        Samp: ChunkDownsampler<N, Usr::Array, Usr::Array>,
    {
        // PERF: Unforunately we have to remove the chunk and put it back to satisfy the borrow checker.
        if let Some(src_node) = self.pop_node_dangling(src_chunk_key) {
            if let Some(src_chunk) = &src_node.user_chunk {
                self.downsample_external_into_self(sampler, src_chunk_key, src_chunk);
            } else {
                self.downsample_ambient_into_self(src_chunk_key);
            }
            self.write_node_dangling(src_chunk_key, src_node);
        } else {
            self.downsample_ambient_into_self(src_chunk_key)
        }
    }

    /// Downsamples all descendants of `ancestor_chunk`, starting at `min_src_lod` and working up the tree.
    ///
    /// The newly created chunks will be passed to the `chunk_rx` callback.
    pub fn downsample_descendants_into_new_chunks<Samp>(
        &self,
        sampler: &Samp,
        ancestor_key: ChunkKey<N>,
        min_src_lod: u8,
        chunk_rx: impl FnMut(ChunkKey<N>, Usr),
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, Usr::Array>,
    {
        self.downsample_descendants_into_new_custom_chunks(
            sampler,
            ancestor_key,
            min_src_lod,
            chunk_rx,
            |extent| self.builder.new_ambient(extent),
        );
    }

    /// Downsamples all descendants of `ancestor_chunk`, starting at `min_src_lod` and working up the tree. Supports a custom
    /// chunk factory `make_ambient`.
    ///
    /// The newly created chunks will be passed to the `chunk_rx` callback.
    pub fn downsample_descendants_into_new_custom_chunks<Samp, DstUsr>(
        &self,
        sampler: &Samp,
        ancestor_key: ChunkKey<N>,
        min_src_lod: u8,
        mut chunk_rx: impl FnMut(ChunkKey<N>, DstUsr),
        mut make_ambient: impl FnMut(ExtentN<N>) -> DstUsr,
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, DstUsr::Array>
            + ChunkDownsampler<N, DstUsr::Array, DstUsr::Array>,
        DstUsr: UserChunk,
        DstUsr::Array: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        assert!(min_src_lod < ancestor_key.lod);
        if let Some(new_chunk) = self.downsample_descendants_into_new_chunks_recursive(
            sampler,
            ancestor_key,
            min_src_lod,
            &mut chunk_rx,
            &mut make_ambient,
        ) {
            chunk_rx(ancestor_key, new_chunk);
        }
    }

    fn downsample_descendants_into_new_chunks_recursive<Samp, DstUsr>(
        &self,
        sampler: &Samp,
        node_key: ChunkKey<N>,
        min_src_lod: u8,
        chunk_rx: &mut impl FnMut(ChunkKey<N>, DstUsr),
        make_ambient: &mut impl FnMut(ExtentN<N>) -> DstUsr,
    ) -> Option<DstUsr>
    where
        Samp: ChunkDownsampler<N, Usr::Array, DstUsr::Array>
            + ChunkDownsampler<N, DstUsr::Array, DstUsr::Array>,
        DstUsr: UserChunk,
        DstUsr::Array: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        if node_key.lod == min_src_lod {
            return None;
        }

        let dst_extent = self.indexer.extent_for_chunk_with_min(node_key.minimum);
        let mut dst_chunk = make_ambient(dst_extent);

        if node_key.lod > min_src_lod {
            self.visit_child_keys(node_key, |child_key, _| {
                let new_child_chunk = self.downsample_descendants_into_new_chunks_recursive(
                    sampler,
                    child_key,
                    min_src_lod,
                    chunk_rx,
                    make_ambient,
                );

                // Do a post-order traversal so we can start sampling from the bottom of the tree and work our way up.
                self.downsample_from_src_into_external(
                    sampler,
                    min_src_lod,
                    child_key,
                    new_child_chunk.as_ref(),
                    dst_chunk.array_mut(),
                );

                if let Some(new_chunk) = new_child_chunk {
                    chunk_rx(child_key, new_chunk);
                }
            });
        }

        Some(dst_chunk)
    }

    fn downsample_from_src_into_external<Samp, DstUsr>(
        &self,
        sampler: &Samp,
        min_src_lod: u8,
        src_chunk_key: ChunkKey<N>,
        src_chunk: Option<&DstUsr>,
        dst_array: &mut DstUsr::Array,
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, DstUsr::Array>
            + ChunkDownsampler<N, DstUsr::Array, DstUsr::Array>,
        DstUsr: UserChunk,
        DstUsr::Array: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        let dst = self.indexer.downsample_destination(src_chunk_key);

        if src_chunk_key.lod == min_src_lod {
            if let Some(src_chunk) = self.get_chunk(src_chunk_key) {
                sampler.downsample(src_chunk.array(), dst_array, dst.offset);
            }
        } else if let Some(src_chunk) = src_chunk {
            sampler.downsample(src_chunk.array(), dst_array, dst.offset);
        };
    }

    /// Downsamples all children of `dst_chunk` into `dst_chunk`.
    ///
    /// **WARNING**: Only occupied chunk slots will be downsampled. You will most likely want to initialize `dst_chunk` to all
    /// ambient values.
    pub fn downsample_children_into_external<Samp, Dst>(
        &self,
        sampler: &Samp,
        dst_chunk_key: ChunkKey<N>,
        dst_array: &mut Dst,
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, Dst>,
        Dst: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        self.visit_child_keys(dst_chunk_key, |child_key, _| {
            self.downsample_into_external(sampler, child_key, dst_array);
        })
    }

    /// Downsamples the chunk at `src_chunk_key` into `dst_chunk`, assuming that `dst_chunk` will be stored in the parent node.
    ///
    /// This enables parallel downsampling via write-out-of-place.
    pub fn downsample_into_external<Samp, Dst>(
        &self,
        sampler: &Samp,
        src_chunk_key: ChunkKey<N>,
        dst_chunk: &mut Dst,
    ) where
        Samp: ChunkDownsampler<N, Usr::Array, Dst>,
        Dst: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        let dst = self.indexer.downsample_destination(src_chunk_key);
        if let Some(src_node) = self.get_node(src_chunk_key) {
            if let Some(src_chunk) = &src_node.user_chunk {
                sampler.downsample(src_chunk.array(), dst_chunk, dst.offset);
            } else {
                Self::downsample_ambient_into_external(
                    self.chunk_shape(),
                    self.ambient_value.clone(),
                    dst.offset,
                    dst_chunk,
                );
            }
        } else {
            Self::downsample_ambient_into_external(
                self.chunk_shape(),
                self.ambient_value.clone(),
                dst.offset,
                dst_chunk,
            );
        }
    }

    /// Downsamples `src_chunk` into `src_chunk_key.lod + 1`.
    fn downsample_external_into_self<Samp, Src>(
        &mut self,
        sampler: &Samp,
        src_chunk_key: ChunkKey<N>,
        src_chunk: &Src,
    ) where
        Src: UserChunk,
        Samp: ChunkDownsampler<N, Src::Array, Usr::Array>,
    {
        let dst = self.indexer.downsample_destination(src_chunk_key);
        let dst_chunk = self.get_mut_chunk_or_insert_ambient(dst.chunk_key);
        sampler.downsample(src_chunk.array(), dst_chunk.array_mut(), dst.offset);
    }

    /// Downsamples an ambient chunk into `lod + 1`. This simply fills the destination extent with the ambient value.
    fn downsample_ambient_into_self(&mut self, src_chunk_key: ChunkKey<N>) {
        let ambient = self.ambient_value.clone();
        let chunk_shape = self.chunk_shape();
        let dst = self.indexer.downsample_destination(src_chunk_key);
        let dst_chunk = self.get_mut_chunk_or_insert_ambient(dst.chunk_key);
        let dst_array = dst_chunk.array_mut();
        Self::downsample_ambient_into_external(chunk_shape, ambient, dst.offset, dst_array);
    }

    /// Downsamples an ambient chunk into `lod + 1`. This simply fills the destination extent with the ambient value.
    fn downsample_ambient_into_external<Dst>(
        chunk_shape: PointN<N>,
        ambient: T,
        dst_offset: Local<N>,
        dst_chunk: &mut Dst,
    ) where
        Dst: FillExtent<N, Item = T> + IndexedArray<N>,
    {
        let dst_extent = ExtentN::from_min_and_shape(
            dst_chunk.extent().minimum + dst_offset.0,
            chunk_shape >> 1,
        );
        dst_chunk.fill_extent(&dst_extent, ambient);
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
        lod0.lod_view_mut(0).fill_extent(&lod0_extent, ambient);

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
        lodn.downsample_extent_into_self_with_lod0(
            get_lod0_chunk,
            &SdfMeanDownsampler,
            0,
            3,
            lod0_extent,
        );
    }
}
