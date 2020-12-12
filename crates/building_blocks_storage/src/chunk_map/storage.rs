mod compression;
mod hash_map;
mod serialization;

pub use compression::*;
pub use hash_map::*;
pub use serialization::*;

use super::Chunk;

use building_blocks_core::prelude::*;

pub trait ChunkStorage<'a, N, T, M>
where
    Chunk<N, T, M>: 'a,
{
    /// Borrow the `Chunk` at `key`.
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, M>>;

    /// Mutably borrow the `Chunk` at `key`.
    fn get_mut(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, M>>;

    /// Mutably borrow the `Chunk` at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M>;

    /// Insert `chunk` at `key`.
    fn insert(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) -> Option<Chunk<N, T, M>>;

    type KeyIter: Iterator<Item = &'a PointN<N>>;

    /// Iterate over all occupied chunk keys.
    fn iter_keys(&'a self) -> Self::KeyIter;

    type ChunkIter: Iterator<Item = (&'a PointN<N>, &'a Chunk<N, T, M>)>;

    /// Iterate over all occupied chunks and their keys.
    fn iter_chunks(&'a self) -> Self::ChunkIter;

    type IntoChunkIter: Iterator<Item = (PointN<N>, Chunk<N, T, M>)>;

    /// Consume self and iterate over all occupied chunks and their keys.
    fn into_iter_chunks(self) -> Self::IntoChunkIter;
}
