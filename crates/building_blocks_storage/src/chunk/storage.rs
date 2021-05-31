pub mod compressible;
pub mod compressible_reader;
pub mod hash_map;

pub use compressible::*;
pub use compressible_reader::*;
pub use hash_map::*;

use building_blocks_core::prelude::*;

use auto_impl::auto_impl;
use serde::{Deserialize, Serialize};

/// The key for a chunk at a particular level of detail.
#[derive(Clone, Copy, Debug, Deserialize, Hash, Eq, PartialEq, Serialize)]
pub struct ChunkKey<N> {
    /// The minimum point of the chunk.
    pub minimum: PointN<N>,
    /// The level of detail. From highest resolution at 0 to lowest resolution at MAX_LOD.
    pub lod: u8,
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

/// Methods for reading chunks from storage.
#[auto_impl(&, &mut)]
pub trait ChunkReadStorage<N, Ch> {
    /// Borrow the chunk at `key`.
    fn get(&self, key: ChunkKey<N>) -> Option<&Ch>;
}

/// Methods for writing chunks from storage.
#[auto_impl(&mut)]
pub trait ChunkWriteStorage<N, Ch> {
    /// Mutably borrow the chunk at `key`.
    fn get_mut(&mut self, key: ChunkKey<N>) -> Option<&mut Ch>;

    /// Mutably borrow the chunk at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Ch,
    ) -> &mut Ch;

    /// Replace the chunk at `key` with `chunk`, returning the old value.
    fn replace(&mut self, key: ChunkKey<N>, chunk: Ch) -> Option<Ch>;

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value.
    fn write(&mut self, key: ChunkKey<N>, chunk: Ch);

    /// Removes and drops the chunk at `key`.
    fn delete(&mut self, key: ChunkKey<N>);

    /// Removes and returns the chunk at `key`.
    fn pop(&mut self, key: ChunkKey<N>) -> Option<Ch>;
}

#[auto_impl(&, &mut)]
pub trait IterChunkKeys<'a, N>
where
    ChunkKey<N>: 'a,
{
    type Iter: Iterator<Item = &'a ChunkKey<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
