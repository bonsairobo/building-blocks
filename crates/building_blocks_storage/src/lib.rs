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
//!   - [Array](crate::Array): N-dimensional, single resolution, bounded, dense array
//!   - [ChunkMap](crate::ChunkMap): N-dimensional, multiple resolution, unbounded, sparse array
//!     - Backed by generic chunk storage, with `HashMap` or `CompressibleChunkStorage` implementations
//!
//! Then there are "meta" lattice maps that provide some extra utility:
//!   - [TransformMap](crate::TransformMap): a wrapper of any kind of lattice map that performs an arbitrary transformation
//!   - [Func](crate::Func): some lattice map traits are implemented for closures (like SDFs)
//!
//! For hierarchical indexing and level of detail:
//!   - [OctreeSet](crate::OctreeSet): bounded bitset of points
//!   - [ChunkedOctreeSet](crate::ChunkedOctreeSet): unbounded bitset of points
//!   - [OctreeChunkIndex](crate::OctreeChunkIndex): just a `ChunkedOctreeSet` that tracks chunks and provides clipmap functionality

#[macro_use]
pub mod access_traits;
pub mod array;
pub mod caching;
pub mod chunk;
pub mod compression;
pub mod func;
pub mod multi_ptr;
pub mod octree;
pub mod signed_distance;
pub mod transform_map;

#[cfg(feature = "sled")]
pub mod database;

/// Used in many generic algorithms to check if a voxel is considered empty.
pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

impl IsEmpty for bool {
    fn is_empty(&self) -> bool {
        !*self
    }
}

// Hash types to use for small keys like `PointN`.
pub type SmallKeyHashMap<K, V> = ahash::AHashMap<K, V>;
pub type SmallKeyHashSet<K> = ahash::AHashSet<K>;
pub type SmallKeyBuildHasher = ahash::RandomState;

pub mod prelude {
    pub use super::{
        array::{Local, Stride},
        chunk::{
            AmbientExtent, Chunk, ChunkDownsampler, ChunkKey, ChunkKey2, ChunkKey3,
            ChunkMapBuilder, ChunkReadStorage, ChunkUnits, ChunkWriteStorage, IterChunkKeys,
            LocalChunkCache, LocalChunkCache2, LocalChunkCache3, PointDownsampler,
            SdfMeanDownsampler,
        },
        compression::{BytesCompression, Compressed, Compression, FromBytesCompression},
        func::Func,
        octree::{
            ChunkedOctreeSet, ClipMapConfig3, ClipMapUpdate3, LodChunkUpdate3, OctreeChunkIndex,
            OctreeNode, OctreeSet, VisitStatus,
        },
        signed_distance::{Sd16, Sd8, SignedDistance},
        transform_map::TransformMap,
        IsEmpty,
    };

    pub use super::access_traits::*;
    pub use super::array::compression::multichannel_aliases::*;
    pub use super::array::multichannel_aliases::*;
    pub use super::chunk::map::multichannel_aliases::*;
    pub use super::chunk::storage::compressible::multichannel_aliases::*;
    pub use super::chunk::storage::compressible_reader::multichannel_aliases::*;
    pub use super::chunk::storage::hash_map::multichannel_aliases::*;

    #[cfg(feature = "lz4")]
    pub use super::compression::Lz4;
    #[cfg(feature = "snap")]
    pub use super::compression::Snappy;
    #[cfg(feature = "sled")]
    pub use super::database::{ChunkDb, ChunkDb2, ChunkDb3};
}

/// Includes all of `prelude` plus the extra-generic types and internal traits used for library development.
pub mod dev_prelude {
    pub use super::prelude::*;

    pub use super::{
        array::{
            channels::{Channel, Channels, FastChannelsCompression},
            compression::FastArrayCompression,
            Array, IndexedArray,
        },
        chunk::{
            ChunkHashMap, ChunkMap, ChunkMap2, ChunkMap3, ChunkMapBuilderNxM,
            CompressibleChunkMapReader, CompressibleChunkStorage, CompressibleChunkStorageReader,
        },
        SmallKeyHashMap, SmallKeyHashSet,
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

#[cfg(test)]
mod test_utilities;
