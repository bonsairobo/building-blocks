pub mod compressible;
pub mod hash_map;

pub use compressible::*;
pub use hash_map::*;

use building_blocks_core::prelude::*;

use auto_impl::auto_impl;
use serde::{Deserialize, Serialize};

/// The key for a chunk at a particular level of detail.
#[derive(Debug, Deserialize, Hash, Eq, Serialize)]
pub struct ChunkKey<N> {
    /// The minimum point of the chunk.
    pub minimum: PointN<N>,
    /// The level of detail. From highest resolution at 0 to lowest resolution at MAX_LOD.
    pub lod: u8,
}

// A few of these traits could be derived. But it seems that derive will not help the compiler infer trait bounds as well.

impl<N> Clone for ChunkKey<N>
where
    PointN<N>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            minimum: self.minimum.clone(),
            lod: self.lod.clone(),
        }
    }
}
impl<N> Copy for ChunkKey<N> where PointN<N>: Copy {}

impl<N> PartialEq for ChunkKey<N>
where
    PointN<N>: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.minimum == other.minimum && self.lod == other.lod
    }
}

/// A 2-dimensional `ChunkKey`.
pub type ChunkKey2 = ChunkKey<[i32; 2]>;
/// A 3-dimensional `ChunkKey`.
pub type ChunkKey3 = ChunkKey<[i32; 3]>;

impl<N> ChunkKey<N> {
    pub fn new(lod: u8, chunk_minimum: PointN<N>) -> Self {
        Self {
            lod,
            minimum: chunk_minimum,
        }
    }
}

/// Methods for reading and writing chunks from/to storage.
pub trait ChunkStorage<N> {
    type Chunk;

    /// Borrow the chunk at `key`.
    fn get(&self, key: ChunkKey<N>) -> Option<&Self::Chunk>;

    /// Mutably borrow the chunk at `key`.
    fn get_mut(&mut self, key: ChunkKey<N>) -> Option<&mut Self::Chunk>;

    /// Mutably borrow the chunk at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Self::Chunk,
    ) -> &mut Self::Chunk;

    /// Replace the chunk at `key` with `chunk`, returning the old value.
    fn replace(&mut self, key: ChunkKey<N>, chunk: Self::Chunk) -> Option<Self::Chunk>;

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value.
    fn write(&mut self, key: ChunkKey<N>, chunk: Self::Chunk);

    /// Removes and drops the chunk at `key`.
    fn delete(&mut self, key: ChunkKey<N>);

    /// Removes and returns the chunk at `key`.
    fn pop(&mut self, key: ChunkKey<N>) -> Option<Self::Chunk>;
}

#[auto_impl(&, &mut)]
pub trait IterChunkKeys<'a, N>
where
    ChunkKey<N>: 'a,
{
    type Iter: Iterator<Item = &'a ChunkKey<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
