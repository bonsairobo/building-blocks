pub mod compressible;
pub mod hash_map;

pub use compressible::*;
pub use hash_map::*;

use super::{ChunkNode, NodeState};

use building_blocks_core::prelude::*;

use auto_impl::auto_impl;

/// Methods for reading and writing chunks from/to storage.
///
/// Depending on the implementation, any method that fetches a `Self::Chunk` might trigger decompression.
pub trait ChunkStorage<N> {
    type Chunk;

    /// The raw representation of a chunk while in storage. Might be compressed in some implementations.
    type ChunkRepr;

    /// Returns true iff storage contains chunk data (not just a node) for `key`.
    fn contains_chunk(&self, key: PointN<N>) -> bool;

    /// Borrow the node at `key`.
    fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>>;

    /// Mutably borrow the node at `key`.
    fn get_mut_node(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Self::Chunk>>;

    /// Mutably borrow the chunk at `key`. If it doesn't exist, insert the return value of `create_node`.
    fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut ChunkNode<Self::Chunk>;

    /// Replace the chunk at `key` with `node`, returning the old value.
    fn replace_node(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Self::Chunk>>;

    /// Overwrite the node at `key` with `node`. Drops the previous value.
    fn write_node(&mut self, key: PointN<N>, node: ChunkNode<Self::Chunk>);

    /// Overwrite the raw node at `key` with `node`. Drops the previous value.
    fn write_raw_node(&mut self, key: PointN<N>, node: ChunkNode<Self::ChunkRepr>);

    /// Removes and returns the node at `key`.
    fn pop_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::Chunk>>;

    /// Removes and returns the raw node at `key`.
    fn pop_raw_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::ChunkRepr>>;

    /// Deletes the chunk (but not the node) at `key`.
    fn delete_chunk(&mut self, key: PointN<N>);

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value. Any previous `NodeState` is preserved.
    fn write_chunk(&mut self, key: PointN<N>, chunk: Self::Chunk);

    fn get_node_state(&self, key: PointN<N>) -> Option<&NodeState>;

    /// Mutably borrow the child mask for the node at `key`. Also returns a bool which is true iff the node has data.
    fn get_mut_node_state(&mut self, key: PointN<N>) -> Option<(&mut NodeState, bool)>;

    fn get_mut_node_state_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut NodeState;
}

#[auto_impl(&, &mut)]
pub trait IterChunkKeys<'a, N>
where
    PointN<N>: 'a,
{
    type Iter: Iterator<Item = &'a PointN<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
