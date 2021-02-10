pub mod chunk_pyramid;
pub mod clipmap;
pub mod sampling;

pub use chunk_pyramid::*;
pub use clipmap::*;
pub use sampling::*;

use building_blocks_core::PointN;

/// The chunk key for a chunk at a particular level of detail.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct LodChunkKey<N> {
    pub chunk_key: PointN<N>,
    pub lod: u8,
}

/// A 2-dimensional `LodChunkKey`.
pub type LodChunkKey2 = LodChunkKey<[i32; 2]>;
/// A 3-dimensional `LodChunkKey`.
pub type LodChunkKey3 = LodChunkKey<[i32; 3]>;
