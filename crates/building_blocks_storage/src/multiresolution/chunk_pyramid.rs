use crate::{
    prelude::*, ArrayNx1, ChunkDownsampler, ChunkMap, ChunkMapBuilder, ChunkMapBuilderNx1,
    ChunkMapNx1, FastArrayCompressionNx1, OctreeChunkIndex, OctreeNode, SmallKeyHashMap,
    VisitStatus,
};

use building_blocks_core::prelude::*;

use std::borrow::Borrow;
use std::fmt::Debug;

/// A set of `ChunkMap`s used as storage for voxels with variable level of detail (LOD).
///
/// All chunks have the same shape, but the voxel size doubles every level of the pyramid.
///
/// The current purpose of this structure is to support downsampling a single voxel data channel. When used with a multichannel
/// `ChunkMap`, LOD0 is managed separately from the mip levels, and it must be provided as an argument to downsampling methods.
///
/// Due to the potentialy ambiguity about whether LOD0 is stored in the pyramid or not, methods will have parameters named
/// either `level` or `lod`. `level` refers to an index within the pyramid. `lod` refers to an actual level of detail,
/// regardless of where LOD0 is stored.
pub struct ChunkPyramid<N, T, Store> {
    // TODO: allow generic builder / multichannel
    levels: Vec<ChunkMapNx1<N, T, Store>>,
    builder: ChunkMapBuilderNx1<N, T>,
}

pub type ChunkPyramid2<T, Store> = ChunkPyramid<[i32; 2], T, Store>;
pub type ChunkPyramid3<T, Store> = ChunkPyramid<[i32; 3], T, Store>;

impl<N, T, Store> ChunkPyramid<N, T, Store> {
    /// The number of levels stored by `self`.
    pub fn num_levels(&self) -> u8 {
        self.levels.len() as u8
    }

    pub fn levels_slice(&self) -> &[ChunkMapNx1<N, T, Store>] {
        &self.levels[..]
    }

    /// Borrow a level with zero-based index `level`.
    pub fn level(&self, level: u8) -> &ChunkMapNx1<N, T, Store> {
        &self.levels[level as usize]
    }

    /// Mutably borrow a level with zero-based index `level`.
    pub fn level_mut(&mut self, level: u8) -> &mut ChunkMapNx1<N, T, Store> {
        &mut self.levels[level as usize]
    }
}

impl<N, T, Store> ChunkPyramid<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Store: ChunkWriteStorage<N, ArrayNx1<N, T>>,
{
    /// Construct a new `ChunkPyramid` with height `num_levels`.
    pub fn new(
        builder: ChunkMapBuilderNx1<N, T>,
        storage_factory: impl Fn() -> Store,
        num_levels: u8,
    ) -> Self
    where
        ChunkMapBuilderNx1<N, T>: Clone,
    {
        let mut levels = Vec::with_capacity(num_levels as usize);
        levels.resize_with(num_levels as usize, || {
            builder.clone().build_with_write_storage(storage_factory())
        });

        Self { levels, builder }
    }

    pub fn chunk_shape(&self) -> PointN<N> {
        self.builder.chunk_shape()
    }

    pub fn ambient_value(&self) -> T {
        self.builder.ambient_value()
    }

    /// Downsamples the chunk at `src_level` and `src_chunk_key` into the specified destination level `dst_level`.
    pub fn downsample_chunk<Samp>(
        &mut self,
        sampler: &Samp,
        src_level: u8,
        src_chunk_key: PointN<N>,
        dst_level: u8,
    ) where
        Samp: ChunkDownsampler<N, T, ArrayNx1<N, T>>,
        ArrayNx1<N, T>: ForEachMutPtr<N, (), Item = *mut T>,
    {
        let Self { levels, builder } = self;

        let chunk_shape = builder.chunk_shape();

        // We don't strictly need to get_mut_chunk for src_chunks, but it's easier than supporting generic ChunkReadStorage when
        // self is already mutably borrowed.
        let [src_chunks, dst_chunks] = two_elems_mut(levels, src_level, dst_level);

        let lod_delta = dst_level - src_level;

        if let Some(src_chunk) = src_chunks.get_mut_chunk(src_chunk_key) {
            downsample_chunk_into_map(
                sampler,
                chunk_shape,
                src_chunk_key,
                src_chunk,
                lod_delta,
                dst_chunks,
            );
        } else {
            Self::downsample_ambient(
                builder.ambient_value(),
                chunk_shape,
                src_chunk_key,
                lod_delta,
                dst_chunks,
            )
        }
    }

    /// The same as `downsample_chunk`, but using a source chunk from `lod0_chunks`. Note that `dst_lod` is NOT treated as an
    /// index into `self.levels`; it is essentially `level + 1`, since LOD0 is not owned by `self`.
    pub fn downsample_lod0_chunk<Samp, Lod0Ch, Lod0ChBorrow>(
        &mut self,
        get_lod0_chunk: impl Fn(PointN<N>) -> Option<Lod0Ch>,
        sampler: &Samp,
        src_chunk_key: PointN<N>,
        dst_lod: u8,
    ) where
        // Need to get tricky here to support both chunk references and TransformMaps returned from a closure.
        Lod0Ch: Borrow<Lod0ChBorrow>,
        Samp: ChunkDownsampler<N, T, Lod0ChBorrow>,
        ArrayNx1<N, T>: ForEachMutPtr<N, (), Item = *mut T>,
    {
        assert!(dst_lod > 0);
        let dst_level = dst_lod - 1;
        let lod_delta = dst_lod;

        let Self { levels, builder } = self;

        let chunk_shape = builder.chunk_shape();

        let dst_chunks = &mut levels[dst_level as usize];

        if let Some(src_chunk) = get_lod0_chunk(src_chunk_key) {
            downsample_chunk_into_map(
                sampler,
                chunk_shape,
                src_chunk_key,
                src_chunk.borrow(),
                lod_delta,
                dst_chunks,
            );
        } else {
            Self::downsample_ambient(
                builder.ambient_value(),
                chunk_shape,
                src_chunk_key,
                lod_delta,
                dst_chunks,
            )
        }
    }

    /// Fill the destination samples with the ambient value.
    fn downsample_ambient(
        ambient_value: T,
        chunk_shape: PointN<N>,
        src_chunk_key: PointN<N>,
        lod_delta: u8,
        dst_chunks: &mut ChunkMapNx1<N, T, Store>,
    ) where
        ArrayNx1<N, T>: ForEachMutPtr<N, (), Item = *mut T>,
    {
        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_chunk_key, lod_delta);
        let dst_chunk = dst_chunks.get_mut_chunk_or_insert_ambient(dst.dst_chunk_key);
        let dst_extent = ExtentN::from_min_and_shape(
            dst_chunk.extent().minimum + dst.dst_offset.0,
            chunk_shape >> 1,
        );
        dst_chunk.fill_extent(&dst_extent, ambient_value);
    }
}

impl<T, Store> ChunkPyramid3<T, Store>
where
    T: Clone,
    Store: ChunkWriteStorage<[i32; 3], Array3x1<T>>,
{
    /// Downsamples all chunks that both:
    ///   1. overlap `extent`
    ///   2. are present in `index`
    pub fn downsample_chunks_with_index<Samp>(
        &mut self,
        index: &OctreeChunkIndex,
        sampler: &Samp,
        extent: &Extent3i,
    ) where
        Samp: ChunkDownsampler<[i32; 3], T, Array3x1<T>>,
    {
        let chunk_shape = self.chunk_shape();
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);

        let chunk_space_extent =
            Extent3i::from_min_and_shape(extent.minimum >> chunk_log2, extent.shape >> chunk_log2);

        index
            .superchunk_octrees
            .visit_octrees(extent, &mut |octree| {
                // Post-order is important to make sure we start downsampling at LOD 0.
                octree.visit_all_octants_for_extent_in_postorder(
                    &chunk_space_extent,
                    &mut |node: &OctreeNode| {
                        let src_lod = node.octant().power();
                        if src_lod < self.num_levels() - 1 {
                            let dst_lod = src_lod + 1;
                            let src_chunk_key =
                                (node.octant().minimum() << chunk_log2) >> src_lod as i32;
                            self.downsample_chunk(sampler, src_lod, src_chunk_key, dst_lod);
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
        Samp: ChunkDownsampler<[i32; 3], T, Array3x1<T>>
            + ChunkDownsampler<[i32; 3], T, Lod0ChBorrow>,
    {
        let chunk_shape = self.chunk_shape();
        let chunk_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);

        let chunk_space_extent =
            Extent3i::from_min_and_shape(extent.minimum >> chunk_log2, extent.shape >> chunk_log2);

        index
            .superchunk_octrees
            .visit_octrees(extent, &mut |octree| {
                // Post-order is important to make sure we start downsampling at LOD 0.
                octree.visit_all_octants_for_extent_in_postorder(
                    &chunk_space_extent,
                    &mut |node: &OctreeNode| {
                        let src_lod = node.octant().power();
                        if src_lod < self.num_levels() - 1 {
                            let dst_lod = src_lod + 1;
                            let src_chunk_key =
                                (node.octant().minimum() << chunk_log2) >> src_lod as i32;

                            if src_lod == 0 {
                                self.downsample_lod0_chunk(
                                    &get_lod0_chunk,
                                    sampler,
                                    src_chunk_key,
                                    dst_lod,
                                );
                            } else {
                                self.downsample_chunk(sampler, src_lod, src_chunk_key, dst_lod);
                            }
                        }

                        VisitStatus::Continue
                    },
                );
            });
    }
}

fn two_elems_mut<T>(levels: &mut [T], level_a: u8, level_b: u8) -> [&mut T; 2] {
    assert!(level_b > level_a);

    let (head, tail) = levels.split_at_mut(level_b as usize);
    let map_a = &mut head[level_a as usize];
    let map_b = &mut tail[level_b as usize - level_a as usize - 1];

    [map_a, map_b]
}

/// Downsamples all of `src_chunk` into the chunks of `dst_chunks` at level `dst_lod`.
pub fn downsample_chunk_into_map<N, T, Samp, SrcCh, B, DstStore>(
    sampler: &Samp,
    chunk_shape: PointN<N>,
    src_chunk_key: PointN<N>,
    src_chunk: &SrcCh,
    lod_delta: u8,
    dst_chunks: &mut ChunkMap<N, T, B, DstStore>,
) where
    PointN<N>: IntegerPoint<N>,
    Samp: ChunkDownsampler<N, T, SrcCh>,
    B: ChunkMapBuilder<N, T, Chunk = ArrayNx1<N, T>>,
    DstStore: ChunkWriteStorage<N, B::Chunk>,
{
    let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_chunk_key, lod_delta);
    let dst_chunk = dst_chunks.get_mut_chunk_or_insert_ambient(dst.dst_chunk_key);
    sampler.downsample(src_chunk, dst_chunk, dst.dst_offset, lod_delta);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DownsampleDestination<N> {
    dst_chunk_key: PointN<N>,
    dst_offset: Local<N>,
}

impl<N> DownsampleDestination<N>
where
    PointN<N>: IntegerPoint<N>,
{
    /// When downsampling a chunk at level `N`, the samples are used at the returned destination within level `N + level_delta`
    /// in the clipmap.
    fn for_source_chunk(chunk_shape: PointN<N>, src_chunk_key: PointN<N>, lod_delta: u8) -> Self {
        let lod_delta = lod_delta as i32;
        let chunk_shape_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        let level_up_log2 = chunk_shape_log2 + PointN::fill(lod_delta);
        let level_up_shape = chunk_shape << lod_delta;
        let dst_chunk_key = (src_chunk_key >> level_up_log2) << chunk_shape_log2;
        let offset = src_chunk_key % level_up_shape;
        let dst_offset = Local(offset >> lod_delta);

        Self {
            dst_chunk_key,
            dst_offset,
        }
    }
}

/// A `ChunkPyramid` using `HashMap` as chunk storage.
pub type ChunkHashMapPyramid<N, T> = ChunkPyramid<N, T, SmallKeyHashMap<PointN<N>, ArrayNx1<N, T>>>;
/// A 2-dimensional `ChunkHashMapPyramid`.
pub type ChunkHashMapPyramid2<T> = ChunkHashMapPyramid<[i32; 2], T>;
/// A 3-dimensional `ChunkHashMapPyramid`.
pub type ChunkHashMapPyramid3<T> = ChunkHashMapPyramid<[i32; 3], T>;

/// A `ChunkPyramid` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkPyramid<N, By, T> =
    ChunkPyramid<N, T, CompressibleChunkStorage<N, FastArrayCompressionNx1<N, By, T>>>;
/// 2-dimensional `CompressibleChunkPyramid`.
pub type CompressibleChunkPyramid2<By, T> = CompressibleChunkPyramid<[i32; 2], T, By>;
/// 3-dimensional `CompressibleChunkPyramid`.
pub type CompressibleChunkPyramid3<By, T> = CompressibleChunkPyramid<[i32; 3], T, By>;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sd8, SdfMeanDownsampler, TransformMap};

    #[test]
    fn downsample_destination_for_one_level_up() {
        let chunk_shape = Point3i::fill(16);
        let level_delta = 1;

        let src_key = chunk_shape;
        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_key: Point3i::ZERO,
                dst_offset: Local(chunk_shape / 2),
            }
        );

        let src_key = 2 * chunk_shape;
        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_key: chunk_shape,
                dst_offset: Local(Point3i::ZERO),
            }
        );
    }

    #[test]
    fn downsample_destination_for_two_levels_up() {
        let chunk_shape = Point3i::fill(16);
        let level_delta = 2;

        let src_key = 3 * chunk_shape;
        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_key: Point3i::ZERO,
                dst_offset: Local(3 * chunk_shape / 4),
            }
        );

        let src_key = 4 * chunk_shape;
        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_key: chunk_shape,
                dst_offset: Local(Point3i::ZERO),
            }
        );
    }

    #[test]
    fn downsample_multichannel_chunks_with_index() {
        let num_lods = 6;
        let chunk_shape = Point3i::fill(16);
        let superchunk_shape = Point3i::fill(512);

        let lod0_extent =
            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(2)) * chunk_shape;

        // Build a multichannel chunk map for LOD0.
        let ambient = (Sd8::ONE, 'a');
        let lod0_builder = ChunkMapBuilder3x2::new(chunk_shape, ambient);
        let mut lod0 = lod0_builder.build_with_hash_map_storage();
        lod0.fill_extent(&lod0_extent, ambient);

        let pyramid_builder = ChunkMapBuilder3x1::new(chunk_shape, Sd8::ONE);
        let mut pyramid = ChunkPyramid3::new(pyramid_builder, || SmallKeyHashMap::new(), num_lods);

        let index = OctreeChunkIndex::index_chunk_map(superchunk_shape, &lod0);

        // Since we're downsampling multichannel chunks, we need to project them onto the one channel that we're downsampling.
        let get_lod0_chunk = |p| {
            lod0.get_chunk(p)
                .map(|chunk| TransformMap::new(chunk, |(sd, _letter): (Sd8, char)| sd))
        };

        pyramid.downsample_chunks_with_lod0_and_index(
            get_lod0_chunk,
            &index,
            &SdfMeanDownsampler,
            &lod0_extent,
        );
    }
}
