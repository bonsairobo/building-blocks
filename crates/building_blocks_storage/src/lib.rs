//! Various types of storage for "lattice maps," functions defined on N-dimensional integer
//! lattices.
//!
//! The core storage types are:
//!   - `ArrayN`: N-dimensional, dense array
//!   - `ChunkMap`: N-dimensional, sparse array
//!
//! Then there are "meta" lattice maps that provide some extra utility:
//!   - `TransformMap`: a wrapper of any kind of lattice map that performs an arbitrary transformation
//!   - `Fn(&PointN<N>)`: some lattice map traits are implemented for functions (like SDFs)

pub mod access;
pub mod array;
pub mod array2;
pub mod array3;
pub mod chunk_map;
pub mod func;
pub mod transform_map;

pub use access::{copy_extent, ForEach, ForEachMut, Get, GetMut, ReadExtent, WriteExtent};
#[cfg(feature = "compress-lz4")]
pub use array::FastLz4;
#[cfg(feature = "compress-snappy")]
pub use array::FastSnappy;
pub use array::{Array, ArrayN, Local, Stride};
pub use array2::Array2;
pub use array3::Array3;
pub use chunk_map::{
    Chunk, Chunk2, Chunk3, ChunkMap, ChunkMap2, ChunkMap3, ChunkMapReader, ChunkMapReader2,
    ChunkMapReader3, LocalChunkCache, LocalChunkCache2, LocalChunkCache3, SerializableChunkMap,
    SerializableChunkMap2, SerializableChunkMap3,
};
pub use transform_map::TransformMap;

/// Used in many generic algorithms to check if a voxel is considered empty.
pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

pub mod prelude {
    pub use super::{
        copy_extent, Array, Array2, Array3, ArrayN, Chunk2, Chunk3, ChunkMap2, ChunkMap3,
        ChunkMapReader2, ChunkMapReader3, Compressible, Decompressible, ForEach,
        ForEachMut, Get, GetMut, IsEmpty, Local, LocalChunkCache2, LocalChunkCache3, ReadExtent,
        Stride, TransformMap, WriteExtent,
    };
    #[cfg(feature = "compress-lz4")]
    pub use super::FastLz4;
    #[cfg(feature = "compress-snappy")]
    pub use super::FastSnappy;
}

pub use compressible_map::{self, Compressible, Decompressible};
