#![allow(
    clippy::type_complexity,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    nonstandard_style,
    future_incompatible
)]
#![warn(clippy::doc_markdown)]
#![doc = include_str!("crate_doc.md")]

#[macro_use]
pub mod access_traits;
pub mod array;
pub mod bitset;
pub mod caching;
pub mod chunk_tree;
pub mod compression;
pub mod func;
#[doc(hidden)]
pub mod multi_ptr;
pub mod octree_set;
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

#[doc(hidden)]
pub mod prelude {
    pub use super::{
        array::{IndexedArray, Local, Stride},
        bitset::{AtomicBitset8, Bitset8},
        chunk_tree::{
            chunk_lod0_bounding_sphere, AmbientExtent, ChunkDownsampler, ChunkIndexer2,
            ChunkIndexer3, ChunkKey, ChunkKey2, ChunkKey3, ChunkNode, ChunkStorage,
            ChunkTreeBuilder, ChunkTreeConfig, ChunkUnits, ClipmapSlot2, ClipmapSlot3,
            IterChunkKeys, LodChange2, LodChange3, PointDownsampler, SdfMeanDownsampler, UserChunk,
        },
        compression::{
            BincodeCompression, BytesCompression, Compressed, Compression, FromBytesCompression,
        },
        func::Func,
        octree_set::{OctreeNode, OctreeSet, OctreeVisitor, VisitStatus},
        signed_distance::{Sd16, Sd8, SignedDistance},
        transform_map::TransformMap,
        IsEmpty,
    };

    pub use super::access_traits::*;
    pub use super::array::compression::multichannel_aliases::*;
    pub use super::array::multichannel_aliases::*;
    pub use super::chunk_tree::builder::multichannel_aliases::*;
    pub use super::chunk_tree::storage::compressible::multichannel_aliases::*;
    pub use super::chunk_tree::storage::hash_map::multichannel_aliases::*;

    #[cfg(feature = "lz4")]
    pub use super::compression::Lz4;
    #[cfg(feature = "snap")]
    pub use super::compression::Snappy;
    #[cfg(feature = "sled")]
    pub use super::database::{
        ChunkDb, ChunkDb2, ChunkDb3, Delta, DeltaBatch, DeltaBatchBuilder, ReadResult,
        ReadableChunkDb,
    };
    #[cfg(feature = "sled-snapshots")]
    pub use super::database::{VersionedChunkDb, VersionedChunkDb2, VersionedChunkDb3};
}

/// Includes all of `prelude` plus the extra-generic types and internal traits used for library development.
#[doc(hidden)]
pub mod dev_prelude {
    pub use super::prelude::*;

    pub use super::{
        array::{
            channels::{Channel, Channels, FastChannelsCompression},
            compression::FastArrayCompression,
            Array, IndexedArray,
        },
        chunk_tree::{
            ChunkIndexer, ChunkTree, ChunkTree2, ChunkTree3, ChunkTreeBuilderNxM,
            CompressibleChunkStorage, HashMapChunkTree,
        },
        SmallKeyHashMap, SmallKeyHashSet,
    };

    #[cfg(feature = "sled")]
    pub use super::database::DatabaseKey;
}

#[cfg(feature = "dot_vox")]
pub mod dot_vox_conversions;
#[cfg(feature = "image")]
pub mod image_conversions;
#[cfg(feature = "vox-format")]
pub mod vox_format;

#[cfg(test)]
mod test_utilities;
