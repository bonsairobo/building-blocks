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
pub mod octree;
pub mod transform_map;

pub use access::{copy_extent, ForEach, ForEachMut, Get, GetMut, ReadExtent, WriteExtent};
pub use array::{
    Array, Array2, Array3, ArrayN, FastArrayCompression, FastCompressedArray, Local, Stride,
};
pub use chunk_map::{
    Chunk, Chunk2, Chunk3, ChunkMap, ChunkMapReader, FastChunkCompression, LocalChunkCache,
    LocalChunkCache2, LocalChunkCache3, SerializableChunkMap,
};
pub use transform_map::TransformMap;

// Only export these aliases when one compression backend is used.
#[cfg(all(feature = "lz4", not(feature = "snappy")))]
pub use chunk_map::conditional_aliases::*;
#[cfg(all(not(feature = "lz4"), feature = "snappy"))]
pub use chunk_map::conditional_aliases::*;

/// Used in many generic algorithms to check if a voxel is considered empty.
pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

pub mod prelude {
    pub use super::{
        copy_extent, Array, Array2, Array3, ArrayN, Chunk2, Chunk3, Compressed, Compression,
        FastArrayCompression, FastChunkCompression, ForEach, ForEachMut, Get, GetMut, IsEmpty,
        Local, LocalChunkCache2, LocalChunkCache3, ReadExtent, Stride, TransformMap, WriteExtent,
    };

    // Only export these aliases when one compression backend is used.
    // Only export these aliases when one compression backend is used.
    #[cfg(all(feature = "lz4", not(feature = "snappy")))]
    pub use super::chunk_map::conditional_aliases::*;
    #[cfg(all(not(feature = "lz4"), feature = "snappy"))]
    pub use super::chunk_map::conditional_aliases::*;

    #[cfg(feature = "lz4")]
    pub use super::Lz4;
    #[cfg(feature = "snappy")]
    pub use super::Snappy;
}

pub use compressible_map::{self, BytesCompression, Compressed, Compression};

#[cfg(feature = "lz4")]
pub use compressible_map::Lz4;
#[cfg(feature = "snappy")]
pub use compressible_map::Snappy;
