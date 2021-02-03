mod caching;
mod compressible;
mod compressible_reader;
mod compression;
mod hash_map;
mod serialization;

pub use caching::*;
pub use compressible::*;
pub use compressible_reader::*;
pub use compression::*;
pub use hash_map::*;
pub use serialization::*;

use super::Chunk;

use building_blocks_core::prelude::*;

/// Methods for reading `Chunk`s from storage.
pub trait ChunkReadStorage<N, T, Meta = ()> {
    /// Borrow the `Chunk` at `key`.
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, Meta>>;
}

/// Methods for writing `Chunk`s from storage.
pub trait ChunkWriteStorage<N, T, Meta = ()> {
    /// Mutably borrow the `Chunk` at `key`.
    fn get_mut(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, Meta>>;

    /// Mutably borrow the `Chunk` at `key`. If it doesn't exist, insert the return value of `create_chunk`.
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, Meta>,
    ) -> &mut Chunk<N, T, Meta>;

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    fn replace(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>) -> Option<Chunk<N, T, Meta>>;

    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    fn write(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>);
}

pub trait IterChunkKeys<'a, N>
where
    PointN<N>: 'a,
{
    type Iter: Iterator<Item = &'a PointN<N>>;

    fn chunk_keys(&'a self) -> Self::Iter;
}
