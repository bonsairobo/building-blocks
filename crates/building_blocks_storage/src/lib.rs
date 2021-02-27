#![allow(
    clippy::type_complexity,
    clippy::needless_collect,
    clippy::too_many_arguments
)]

//! Various types of storage and indexing for voxels in 2 or 3 dimensions.
//!
//! If you need to store signed distance values in your voxels, consider using the `Sd8` and `Sd16` fixed-precision types which
//! implement the `SignedDistance` trait required for smooth meshing.
//!
//! The core storage types are:
//!   - `ArrayN`: N-dimensional, dense array
//!   - `ChunkHashMap`: N-dimensional, sparse array
//!   - `CompressibleChunkMap`: N-dimensional, sparse array with chunk compression
//!
//! Then there are "meta" lattice maps that provide some extra utility:
//!   - `TransformMap`: a wrapper of any kind of lattice map that performs an arbitrary transformation
//!   - `Fn(PointN<N>)`: some lattice map traits are implemented for functions (like SDFs)
//!
//! For multiresolution voxel data, there is an extension of `ChunkMap` called the `ChunkPyramid` which supports generic chunk
//! downsampling via the `ChunkDownsampler` trait.
//!
//! For spatial indexing, there is the bounded `OctreeSet` and corresponding unbounded `ChunkedOctreeSet`. Specifically for
//! indexing chunk keys and interacting with clipmaps, there is an `OctreeChunkIndex`.

#[macro_use]
pub mod access_traits;
pub mod array;
pub mod caching;
pub mod chunk;
pub mod chunk_indexer;
pub mod chunk_map;
pub mod chunk_storage;
pub mod chunked_octree_set;
pub mod compression;
pub mod func;
pub mod multiresolution;
pub mod octree_chunk_index;
pub mod octree_set;
pub mod signed_distance;
pub mod transform_map;

pub use access_traits::*;
pub use array::*;
pub use caching::*;
pub use chunk::*;
pub use chunk_indexer::*;
pub use chunk_map::*;
pub use chunk_storage::*;
pub use chunked_octree_set::*;
pub use compression::*;
pub use multiresolution::*;
pub use octree_chunk_index::*;
pub use octree_set::*;
pub use signed_distance::*;
pub use transform_map::*;

/// Used in many generic algorithms to check if a voxel is considered empty.
pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

impl IsEmpty for bool {
    fn is_empty(&self) -> bool {
        !*self
    }
}

pub mod prelude {
    pub use super::{
        copy_extent, Array, Array2, Array3, ArrayN, Chunk, Chunk2, Chunk3, ChunkHashMap2,
        ChunkHashMap3, ChunkHashMapPyramid2, ChunkHashMapPyramid3, ChunkIndexer, ChunkMap,
        ChunkMap2, ChunkMap3, ChunkMapBuilder, ChunkMapBuilder2, ChunkMapBuilder3, ChunkPyramid2,
        ChunkPyramid3, ChunkReadStorage, ChunkWriteStorage, Compressed, CompressibleChunkMap,
        CompressibleChunkMapReader, CompressibleChunkStorage, CompressibleChunkStorageReader,
        Compression, FastArrayCompression, FastChunkCompression, ForEach, ForEachMut, Get, GetMut,
        GetRef, IsEmpty, IterChunkKeys, Local, LocalChunkCache, LocalChunkCache2, LocalChunkCache3,
        OctreeChunkIndex, OctreeNode, OctreeSet, ReadExtent, SerializableChunks, SignedDistance,
        Stride, TransformMap, WriteExtent,
    };

    #[cfg(feature = "lz4")]
    pub use super::Lz4;
    #[cfg(feature = "snap")]
    pub use super::Snappy;

    #[cfg(any(
        all(feature = "lz4", not(feature = "snap")),
        all(not(feature = "lz4"), feature = "snap"),
    ))]
    pub use super::{
        CompressibleChunkMap2, CompressibleChunkMap3, CompressibleChunkMapReader2,
        CompressibleChunkMapReader3, CompressibleChunkPyramid2, CompressibleChunkPyramid3,
        CompressibleChunkStorage2, CompressibleChunkStorage3, CompressibleChunkStorageReader2,
        CompressibleChunkStorageReader3, MaybeCompressedChunk2, MaybeCompressedChunk3,
        MaybeCompressedChunkRef2, MaybeCompressedChunkRef3,
    };
}

#[cfg(feature = "dot_vox")]
mod dot_vox_conversions;
#[cfg(feature = "dot_vox")]
pub use dot_vox_conversions::*;
#[cfg(feature = "image")]
mod image_conversions;
#[cfg(feature = "image")]
pub use image_conversions::*;
