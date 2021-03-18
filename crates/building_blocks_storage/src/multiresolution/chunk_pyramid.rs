use crate::{
    prelude::*, ArrayIndexer, ArrayNx1, BytesCompression, ChunkDownsampler, ChunkHashMapNx1,
    ChunkMap, ChunkMapBuilder, ChunkMapBuilderNx1, ChunkMapNx1, CompressibleChunkMapNx1,
    FastArrayCompressionNx1, OctreeChunkIndex, OctreeNode, SmallKeyHashMap, VisitStatus,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use std::fmt::Debug;

/// A set of `ChunkMap`s used as storage for voxels with variable level of detail (LOD).
///
/// All chunks have the same shape, but the voxel size doubles every level of the pyramid.
///
/// There is no enforcement of a particular occupancy, allowing you to use this as a cache. Typically you will have some region
/// of highest detail close to a central point. Then as you get further from the center, the detail drops.
pub struct ChunkPyramid<N, T, Store> {
    levels: Vec<ChunkMap<N, T, ChunkMapBuilderNx1<N, T>, Store>>,
}

pub type ChunkPyramid2<T, Store> = ChunkPyramid<[i32; 2], T, Store>;
pub type ChunkPyramid3<T, Store> = ChunkPyramid<[i32; 3], T, Store>;

impl<N, T, Store> ChunkPyramid<N, T, Store> {
    pub fn chunk_shape(&self) -> PointN<N>
    where
        PointN<N>: IntegerPoint<N>,
    {
        self.levels[0].indexer.chunk_shape()
    }

    pub fn num_lods(&self) -> u8 {
        self.levels.len() as u8
    }

    pub fn levels_slice(&self) -> &[ChunkMapNx1<N, T, Store>] {
        &self.levels[..]
    }

    pub fn level(&self, lod: u8) -> &ChunkMapNx1<N, T, Store> {
        &self.levels[lod as usize]
    }

    pub fn level_mut(&mut self, lod: u8) -> &mut ChunkMapNx1<N, T, Store> {
        &mut self.levels[lod as usize]
    }

    pub fn two_levels_mut(&mut self, lod_a: u8, lod_b: u8) -> [&mut ChunkMapNx1<N, T, Store>; 2] {
        assert!(lod_a < lod_b);

        // A trick to borrow mutably two different levels.
        let (head, tail) = self.levels.split_at_mut(lod_b as usize);
        let map_a = &mut head[lod_a as usize];
        let map_b = &mut tail[lod_b as usize - lod_a as usize - 1];

        [map_a, map_b]
    }
}

impl<N, T, Store> ChunkPyramid<N, T, Store>
where
    N: ArrayIndexer<N>,
    PointN<N>: Debug + IntegerPoint<N>,
    ChunkIndexer<N>: Clone,
    T: 'static + Clone,
    ArrayNx1<N, T>: Chunk,
    Store: ChunkWriteStorage<N, ArrayNx1<N, T>>,
{
    pub fn downsample_chunk<Samp>(
        &mut self,
        sampler: &Samp,
        src_chunk_key: PointN<N>,
        src_lod: u8,
        dst_lod: u8,
    ) where
        Samp: ChunkDownsampler<N, T>,
        PointN<N>: Debug,
    {
        assert!(dst_lod > src_lod);
        let [src_map, dst_map] = self.two_levels_mut(src_lod, dst_lod);
        let lod_delta = dst_lod - src_lod;

        let chunk_shape = src_map.indexer.chunk_shape();
        let ambient_value = src_map.builder().ambient_value();

        let dst = DownsampleDestination::for_source_chunk(chunk_shape, src_chunk_key, lod_delta);
        let dst_chunk = dst_map.get_mut_chunk_or_insert_ambient(dst.dst_chunk_key);

        // While not strictly necessary to get_mut here, it is much simpler than trying to enforce generic ChunkReadStorage.
        if let Some(src_chunk) = src_map.get_mut_chunk(src_chunk_key) {
            debug_assert_eq!(src_chunk.extent().shape, chunk_shape);

            sampler.downsample(&src_chunk, dst_chunk, dst.dst_offset, lod_delta);
        } else {
            let dst_extent = ExtentN::from_min_and_shape(
                dst_chunk.extent().minimum + dst.dst_offset.0,
                chunk_shape >> 1,
            );
            dst_chunk.fill_extent(&dst_extent, ambient_value);
        }
    }
}

impl<T, Store> ChunkPyramid3<T, Store>
where
    T: 'static + Clone,
    Array3x1<T>: Chunk,
    Store: ChunkWriteStorage<[i32; 3], Array3x1<T>>,
{
    pub fn downsample_chunks_for_extent_all_lods_with_index<Samp>(
        &mut self,
        index: &OctreeChunkIndex,
        sampler: &Samp,
        extent: &Extent3i,
    ) where
        Samp: ChunkDownsampler<[i32; 3], T>,
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
                        let dst_lod = node.octant().power();
                        if dst_lod > 0 && dst_lod < self.num_lods() {
                            // Destination LOD > 0, so we should downsample all non-empty children.
                            let src_lod = dst_lod - 1;
                            for child_octant_index in 0..8 {
                                if node.child_bitmask() & (1 << child_octant_index) != 0 {
                                    let child_octant = node.octant().child(child_octant_index);
                                    let src_chunk_key =
                                        (child_octant.minimum() << chunk_log2) >> src_lod as i32;
                                    self.downsample_chunk(sampler, src_chunk_key, src_lod, dst_lod);
                                }
                            }
                        }

                        VisitStatus::Continue
                    },
                );
            });
    }
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashMapPyramid<N, T> = ChunkPyramid<N, T, SmallKeyHashMap<PointN<N>, ArrayNx1<N, T>>>;
/// A 2-dimensional `ChunkHashMapPyramid`.
pub type ChunkHashMapPyramid2<T> = ChunkHashMapPyramid<[i32; 2], T>;
/// A 3-dimensional `ChunkHashMapPyramid`.
pub type ChunkHashMapPyramid3<T> = ChunkHashMapPyramid<[i32; 3], T>;

impl<N, T> ChunkHashMapPyramid<N, T>
where
    PointN<N>: Hash + IntegerPoint<N>,
    T: Clone,
    ChunkMapBuilderNx1<N, T>: Clone,
{
    pub fn new(builder: ChunkMapBuilderNx1<N, T>, num_lods: u8) -> Self {
        let mut levels = Vec::with_capacity(num_lods as usize);
        levels.resize_with(num_lods as usize, || {
            builder
                .clone()
                .build_with_write_storage(SmallKeyHashMap::default())
        });

        Self { levels }
    }

    pub fn with_lod0_chunk_map(lod0_chunk_map: ChunkHashMapNx1<N, T>, num_lods: u8) -> Self {
        let mut pyramid = Self::new(lod0_chunk_map.builder().clone(), num_lods);
        *pyramid.level_mut(0) = lod0_chunk_map;

        pyramid
    }
}

/// A `ChunkMap` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkPyramid<N, T, B> =
    ChunkPyramid<N, T, CompressibleChunkStorage<N, FastArrayCompressionNx1<N, T, B>>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        /// 2-dimensional `CompressibleChunkPyramid`.
        pub type CompressibleChunkPyramid2<T, B = $backend> =
            CompressibleChunkPyramid<[i32; 2], T, B>;
        /// 3-dimensional `CompressibleChunkPyramid`.
        pub type CompressibleChunkPyramid3<T, B = $backend> =
            CompressibleChunkPyramid<[i32; 3], T, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);

impl<N, T, B> CompressibleChunkPyramid<N, T, B>
where
    PointN<N>: Hash + IntegerPoint<N>,
    T: 'static + Copy,
    B: BytesCompression + Copy,
    ChunkMapBuilderNx1<N, T>: Clone,
{
    pub fn new(builder: ChunkMapBuilderNx1<N, T>, num_lods: u8, compression: B) -> Self {
        let mut levels = Vec::with_capacity(num_lods as usize);
        levels.resize_with(num_lods as usize, || {
            builder.clone().build_with_write_storage(
                FastCompressibleChunkStorage::with_bytes_compression(compression),
            )
        });

        Self { levels }
    }

    pub fn with_lod0_chunk_map(
        lod0_chunk_map: CompressibleChunkMapNx1<N, T, B>,
        num_lods: u8,
    ) -> Self {
        let mut pyramid = Self::new(
            lod0_chunk_map.builder().clone(),
            num_lods,
            *lod0_chunk_map
                .storage()
                .compression
                .channels_compression()
                .bytes_compression(),
        );
        *pyramid.level_mut(0) = lod0_chunk_map;

        pyramid
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownsampleDestination<N> {
    pub dst_chunk_key: PointN<N>,
    pub dst_offset: Local<N>,
}

pub type DownsampleDestination2 = DownsampleDestination<[i32; 2]>;
pub type DownsampleDestination3 = DownsampleDestination<[i32; 3]>;

impl<N> DownsampleDestination<N>
where
    PointN<N>: IntegerPoint<N>,
{
    /// When downsampling a chunk at level `N`, the samples are used at the returned destination within level `N + level_delta`
    /// in the clipmap.
    pub fn for_source_chunk(
        chunk_shape: PointN<N>,
        src_chunk_key: PointN<N>,
        lod_delta: u8,
    ) -> Self {
        let lod_delta = lod_delta as i32;
        let chunk_shape_log2 = chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32);
        let level_up_log2 = chunk_shape_log2 + PointN::ONES * lod_delta;
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

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_destination_for_one_level_up() {
        let chunk_shape = Point3i::fill(16);
        let level_delta = 1;

        let src_key = chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: Point3i::ZERO,
                dst_offset: Local(chunk_shape / 2),
            }
        );

        let src_key = 2 * chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
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
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: Point3i::ZERO,
                dst_offset: Local(3 * chunk_shape / 4),
            }
        );

        let src_key = 4 * chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: chunk_shape,
                dst_offset: Local(Point3i::ZERO),
            }
        );
    }
}
