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
pub mod chunk_map;
pub mod func;
pub mod transform_map;

pub use access::{copy_extent, ForEach, ForEachMut, Get, GetMut, ReadExtent, WriteExtent};
pub use array::{
    Array, Array2, Array3, ArrayN, FastArrayCompression, FastCompressedArray, Local, Stride,
};
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
        ChunkMapReader2, ChunkMapReader3, Compressed, Compression, FastArrayCompression, ForEach,
        ForEachMut, Get, GetMut, IsEmpty, Local, LocalChunkCache2, LocalChunkCache3, ReadExtent,
        Snappy, Stride, TransformMap, WriteExtent,
    };

    #[cfg(feature = "lz4")]
    pub use super::Lz4;
}

pub use compressible_map::{self, Compressed, Compression, Snappy};

#[cfg(feature = "lz4")]
pub use compressible_map::Lz4;
