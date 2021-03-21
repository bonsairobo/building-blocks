pub mod compressible;
pub mod compressible_reader;
pub mod hash_map;

pub use compressible::*;
pub use compressible_reader::*;
pub use hash_map::*;

use building_blocks_core::prelude::*;

/// Methods for reading chunks from storage.
pub trait ChunkReadStorage<N, Ch> {
    /// Borrow the chunk at `key`.
    fn get(&self, key: PointN<N>) -> Option<&Ch>;
}

/// Methods for writing chunks from storage.
pub trait ChunkWriteStorage<N, Ch> {
    /// Mutably borrow the chunk at `key`.
    fn get_mut(&mut self, key: PointN<N>) -> Option<&mut Ch>;

    /// Mutably borrow the chunk at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Ch,
    ) -> &mut Ch;

    /// Replace the chunk at `key` with `chunk`, returning the old value.
    fn replace(&mut self, key: PointN<N>, chunk: Ch) -> Option<Ch>;

    /// Overwrite the chunk at `key` with `chunk`. Drops the previous value.
    fn write(&mut self, key: PointN<N>, chunk: Ch);
}

pub trait IterChunkKeys<'a, N>
where
    PointN<N>: 'a,
{
    type Iter: Iterator<Item = &'a PointN<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
