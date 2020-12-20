use crate::{Chunk, ChunkMap};

use super::{ChunkReadStorage, ChunkWriteStorage, IterChunkKeys};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use std::collections::hash_map;

impl<N, T, Meta> ChunkReadStorage<N, T, Meta> for FnvHashMap<PointN<N>, Chunk<N, T, Meta>>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, Meta>> {
        self.get(key)
    }
}

impl<N, T, Meta> ChunkWriteStorage<N, T, Meta> for FnvHashMap<PointN<N>, Chunk<N, T, Meta>>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get_mut(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, Meta>> {
        self.get_mut(key)
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, Meta>,
    ) -> &mut Chunk<N, T, Meta> {
        self.entry(key).or_insert_with(create_chunk)
    }

    #[inline]
    fn replace(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>) -> Option<Chunk<N, T, Meta>> {
        self.insert(key, chunk)
    }

    #[inline]
    fn write(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>) {
        self.insert(key, chunk);
    }
}

impl<'a, N, T, Meta> IterChunkKeys<'a, N> for FnvHashMap<PointN<N>, Chunk<N, T, Meta>>
where
    PointN<N>: 'a,
    Chunk<N, T, Meta>: 'a,
{
    type Iter = hash_map::Keys<'a, PointN<N>, Chunk<N, T, Meta>>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.keys()
    }
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashMap<N, T, Meta = ()> =
    ChunkMap<N, T, Meta, FnvHashMap<PointN<N>, Chunk<N, T, Meta>>>;
/// A 2-dimensional `ChunkHashMap`.
pub type ChunkHashMap2<T, Meta = ()> = ChunkHashMap<[i32; 2], T, Meta>;
/// A 3-dimensional `ChunkHashMap`.
pub type ChunkHashMap3<T, Meta = ()> = ChunkHashMap<[i32; 3], T, Meta>;
