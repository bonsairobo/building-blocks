use crate::{prelude::*, ArrayIndexer, BytesCompression};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use std::fmt::Debug;

/// A sparse clipmap used for level of detail (LOD).
///
/// This is just a pyramid of `ChunkMap`s. All chunks have the same shape, but the voxel size doubles every level of the
/// pyramid. A `ChunkDownsampler` is used to populate a chunk at a given layer by sampling from multiple chunks at a layer of
/// higher detail.
///
/// There is no enforcement of a particular occupancy, allowing you to use this as a cache. Typically you will have some region
/// of highest detail close to a central point. Then as you get further from the center, the detail drops.
pub struct ChunkClipMap<N, T, Store> {
    levels: Vec<ChunkMap<N, T, (), Store>>,
}

impl<N, T, Store> ChunkClipMap<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
    T: Copy,
    Store: ChunkWriteStorage<N, T, ()>,
{
    pub fn get_level_map(&self, lod: u8) -> &ChunkMap<N, T, (), Store> {
        &self.levels[lod as usize - 1]
    }

    pub fn get_mut_level_map(&mut self, lod: u8) -> &mut ChunkMap<N, T, (), Store> {
        &mut self.levels[lod as usize - 1]
    }

    /// Downsamples `lod0_chunk` from the highest level of detail into `dst_level`, which is some level greater than 0.
    pub fn downsample_lod0<Samp>(
        &mut self,
        sampler: &Samp,
        lod0_chunk_key: PointN<N>,
        lod0_chunk: &ArrayN<N, T>,
        dst_lod: u8,
    ) where
        Samp: ChunkDownsampler<N, T>,
        PointN<N>: Debug,
    {
        assert!(dst_lod as usize <= self.levels.len());
        downsample_lod0(
            sampler,
            lod0_chunk_key,
            lod0_chunk,
            dst_lod,
            self.get_mut_level_map(dst_lod),
        );
    }
}

/// Downsamples `lod0_chunk` from the highest level of detail into the `ChunkMap` at `dst_level`, which is some level greater
/// than 0.
pub fn downsample_lod0<N, T, Meta, Store, Samp>(
    sampler: &Samp,
    lod0_chunk_key: PointN<N>,
    lod0_chunk: &ArrayN<N, T>,
    dst_lod: u8,
    dst_lod_map: &mut ChunkMap<N, T, Meta, Store>,
) where
    PointN<N>: Debug + IntegerPoint<N>,
    T: Copy,
    Meta: Clone,
    Samp: ChunkDownsampler<N, T>,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    assert!(dst_lod > 0);

    debug_assert_eq!(lod0_chunk.extent().shape, dst_lod_map.indexer.chunk_shape());

    let dst =
        DownsampleDestination::for_source_chunk(lod0_chunk.extent().shape, lod0_chunk_key, dst_lod);
    let dst_chunk = dst_lod_map.get_mut_chunk_or_insert_ambient(dst.dst_chunk_key);

    sampler.downsample(lod0_chunk, &mut dst_chunk.array, dst.dst_offset, dst_lod);
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashClipMap<N, T> = ChunkClipMap<N, T, FnvHashMap<PointN<N>, Chunk<N, T, ()>>>;
/// A 2-dimensional `ChunkHashClipMap`.
pub type ChunkHashClipMap2<T> = ChunkHashClipMap<[i32; 2], T>;
/// A 3-dimensional `ChunkHashClipMap`.
pub type ChunkHashClipMap3<T> = ChunkHashClipMap<[i32; 3], T>;

impl<N, T> ChunkHashClipMap<N, T>
where
    PointN<N>: Hash + IntegerPoint<N>,
{
    pub fn with_hash_map_level_stores(builder: ChunkMapBuilder<N, T>, num_levels: usize) -> Self
    where
        ChunkMapBuilder<N, T>: Copy,
    {
        let mut levels = Vec::with_capacity(num_levels);
        levels.resize_with(num_levels, || {
            builder.build_with_write_storage(FnvHashMap::default())
        });

        Self { levels }
    }
}

/// A `ChunkMap` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkClipMap<N, T, B> =
    ChunkClipMap<N, T, CompressibleChunkStorage<N, T, (), B>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        /// 2-dimensional `CompressibleChunkClipMap`.
        pub type CompressibleChunkClipMap2<T, B = $backend> =
            CompressibleChunkClipMap<[i32; 2], T, B>;
        /// 3-dimensional `CompressibleChunkClipMap`.
        pub type CompressibleChunkClipMap3<T, B = $backend> =
            CompressibleChunkClipMap<[i32; 3], T, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);

impl<N, T, B> CompressibleChunkClipMap<N, T, B>
where
    PointN<N>: Hash + IntegerPoint<N>,
    T: Copy,
    B: BytesCompression,
{
    pub fn with_compressible_level_stores(
        builder: ChunkMapBuilder<N, T>,
        num_levels: usize,
        compression: B,
    ) -> Self
    where
        B: Copy,
        ChunkMapBuilder<N, T>: Copy,
    {
        let mut levels = Vec::with_capacity(num_levels);
        levels.resize_with(num_levels, || {
            builder.build_with_write_storage(CompressibleChunkStorage::new(compression))
        });

        Self { levels }
    }
}

pub trait ChunkDownsampler<N, T> {
    /// Samples `src_chunk` in order to write out just a portion of `dst_chunk`, starting at `dst_min`.
    fn downsample(
        &self,
        src_chunk: &ArrayN<N, T>,
        dst_chunk: &mut ArrayN<N, T>,
        dst_min: PointN<N>,
        level_delta: u8,
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownsampleDestination<N> {
    pub dst_chunk_key: PointN<N>,
    pub dst_offset: PointN<N>,
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
        let dst_offset = offset >> lod_delta;

        Self {
            dst_chunk_key,
            dst_offset,
        }
    }
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
        dst_min: PointN<N>,
        lod_delta: u8,
    ) {
        // PERF: this might be faster using Strides

        debug_assert!(lod_delta > 0);
        let lod_delta = lod_delta as i32;

        let sample_shape = src_chunk.extent().shape >> lod_delta;
        debug_assert!(sample_shape > PointN::ZERO);

        for p in ExtentN::from_min_and_shape(PointN::ZERO, sample_shape).iter_points() {
            *dst_chunk.get_mut(&Local(dst_min + p)) = src_chunk.get(&Local(p << lod_delta));
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
        let chunk_shape = PointN([16; 3]);
        let level_delta = 1;

        let src_key = chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: PointN([0; 3]),
                dst_offset: chunk_shape / 2,
            }
        );

        let src_key = 2 * chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: chunk_shape,
                dst_offset: Point3i::ZERO,
            }
        );
    }

    #[test]
    fn downsample_destination_for_two_levels_up() {
        let chunk_shape = PointN([16; 3]);
        let level_delta = 2;

        let src_key = 3 * chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: PointN([0; 3]),
                dst_offset: 3 * chunk_shape / 4,
            }
        );

        let src_key = 4 * chunk_shape;
        let dst = DownsampleDestination3::for_source_chunk(chunk_shape, src_key, level_delta);
        assert_eq!(
            dst,
            DownsampleDestination3 {
                dst_chunk_key: chunk_shape,
                dst_offset: Point3i::ZERO,
            }
        );
    }
}
