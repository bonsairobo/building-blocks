pub mod compressible;
pub mod hash_map;

pub use compressible::*;
pub use hash_map::*;

use super::{ChunkNode, NodeState};

use building_blocks_core::prelude::*;

use auto_impl::auto_impl;
use either::Either;

/// Methods for reading and writing chunk nodes from/to storage.
///
/// Depending on the implementation, any method that fetches a `Self::Chunk` might trigger decompression.
pub trait ChunkStorage<N> {
    /// The data stored for an occupied [`ChunkNode`]. This should probably implement [`UserChunk`](super::UserChunk).
    type Chunk;

    /// The "cold" representation of a chunk, e.g. it may be compressed or live somewhere else entirely.
    ///
    /// Simple implementations of `ChunkStorage` can just have `ColdChunk = Chunk`.
    type ColdChunk;

    /// Inserts `node` at `key` and returns the previous raw node.
    fn insert_node(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Either<Self::Chunk, Self::ColdChunk>>>;

    /// Borrow the node at `key`.
    fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>>;

    /// Borrow the node at `key` in its raw representation.
    fn get_raw_node(
        &self,
        key: PointN<N>,
    ) -> Option<(&NodeState, Either<Option<&Self::Chunk>, &Self::ColdChunk>)>;

    /// Borrow the node state at `key`. The returned `bool` is `true` iff this node has data.
    fn get_node_state(&self, key: PointN<N>) -> Option<(&NodeState, bool)>;

    /// Mutably borrow the node at `key`. Iff `drop_chunk` is `true`, then the chunk on the node is dropped.
    fn get_mut_node(
        &mut self,
        key: PointN<N>,
        drop_chunk: bool,
    ) -> Option<&mut ChunkNode<Self::Chunk>>;

    /// Mutably borrow the node at `key`. If it doesn't exist, insert the return value of `create_node`.
    ///
    /// Iff `drop_chunk` is `true`, then the node's chunk is dropped before being returned. This can be used to prevent
    /// decompression when the chunk isn't needed.
    ///
    /// Returns a `bool` indicating if the node had a chunk, for convenience.
    fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        drop_chunk: bool,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> (&mut ChunkNode<Self::Chunk>, bool);

    /// Mutably borrow the node state at `key`. The returned `bool` is `true` iff this node has data.
    fn get_mut_node_state(&mut self, key: PointN<N>) -> Option<(&mut NodeState, bool)>;

    /// Mutably borrow the node state at `key`. If it doesn't exist, insert the return value of `create_node`. The returned
    /// `bool` is `true` iff this node has data.
    fn get_mut_node_state_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> (&mut NodeState, bool);

    /// Removes and returns the node at `key`.
    fn pop_node(&mut self, key: PointN<N>, drop_chunk: bool) -> Option<ChunkNode<Self::Chunk>>;

    /// Removes and returns the raw node at `key`.
    fn pop_raw_node(
        &mut self,
        key: PointN<N>,
    ) -> Option<ChunkNode<Either<Self::Chunk, Self::ColdChunk>>>;
}

#[auto_impl(&, &mut)]
pub trait IterChunkKeys<'a, N>
where
    PointN<N>: 'a,
{
    type Iter: Iterator<Item = &'a PointN<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
