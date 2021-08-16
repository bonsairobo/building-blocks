pub mod compressible;
pub mod hash_map;

pub use compressible::*;
pub use hash_map::*;

use super::ChunkNode;

use building_blocks_core::prelude::*;

use auto_impl::auto_impl;

/// Methods for reading and writing chunks from/to storage.
pub trait ChunkStorage<N> {
    type Chunk;

    /// Borrow the chunk at `key`.
    fn get(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>>;

    /// Mutably borrow the chunk at `key`.
    fn get_mut(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Self::Chunk>>;

    /// Mutably borrow the chunk at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut ChunkNode<Self::Chunk>;

    /// Replace the chunk at `key` with `chunk`, returning the old value.
    fn replace(
        &mut self,
        key: PointN<N>,
        chunk: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Self::Chunk>>;

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value.
    fn write(&mut self, key: PointN<N>, chunk: ChunkNode<Self::Chunk>);

    /// Removes and drops the chunk at `key`.
    fn delete(&mut self, key: PointN<N>);

    /// Removes and returns the chunk at `key`.
    fn pop(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::Chunk>>;
}

#[auto_impl(&, &mut)]
pub trait IterChunkKeys<'a, N>
where
    PointN<N>: 'a,
{
    type Iter: Iterator<Item = &'a PointN<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
