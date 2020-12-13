#![allow(
    clippy::type_complexity,
    clippy::needless_collect,
    clippy::too_many_arguments
)]

//! Various types of storage for "lattice maps," functions defined on N-dimensional integer lattices.
//!
//! The core storage types are:
//!   - `ArrayN`: N-dimensional, dense array
//!   - `ChunkHashMap`: N-dimensional, sparse array
//!   - `CompressibleChunkMap`: N-dimensional, sparse array with chunk compression
//!
//! Then there are "meta" lattice maps that provide some extra utility:
//!   - `TransformMap`: a wrapper of any kind of lattice map that performs an arbitrary transformation
//!   - `Fn(&PointN<N>)`: some lattice map traits are implemented for functions (like SDFs)

pub mod access;
pub mod array;
pub mod chunk_map;
pub mod compression;
pub mod func;
pub mod octree;
pub mod transform_map;

pub use access::*;
pub use array::*;
pub use chunk_map::*;
pub use compression::*;
pub use octree::*;
pub use transform_map::*;

/// Used in many generic algorithms to check if a voxel is considered empty.
pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

pub mod prelude {
    pub use super::{
        copy_extent, Array, Array2, Array3, ArrayN, Chunk, Chunk2, Chunk3, ChunkHashMap2,
        ChunkHashMap3, ChunkIndexer, ChunkMap, ChunkMap2, ChunkMap3, ChunkMapBuilder,
        ChunkMapBuilder2, ChunkMapBuilder3, ChunkReadStorage, ChunkWriteStorage, Compressed,
        CompressibleChunkMap, CompressibleChunkStorage, CompressibleChunkStorageReader,
        Compression, FastArrayCompression, FastChunkCompression, ForEach, ForEachMut, Get, GetMut,
        IsEmpty, IterChunkKeys, Local, LocalChunkCache, LocalChunkCache2, LocalChunkCache3,
        OctreeSet, ReadExtent, SerializableChunkMap, Stride, TransformMap, WriteExtent,
    };

    pub use super::chunk_map::conditional_aliases::*;

    #[cfg(feature = "lz4")]
    pub use super::Lz4;
    #[cfg(feature = "snap")]
    pub use super::Snappy;
}
