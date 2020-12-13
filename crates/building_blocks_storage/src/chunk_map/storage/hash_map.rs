use crate::{Chunk, ChunkMap};

use super::{ChunkReadStorage, ChunkWriteStorage, IterChunkKeys};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use std::collections::hash_map;

impl<N, T, M> ChunkReadStorage<N, T, M> for FnvHashMap<PointN<N>, Chunk<N, T, M>>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, M>> {
        self.get(key)
    }
}

impl<N, T, M> ChunkWriteStorage<N, T, M> for FnvHashMap<PointN<N>, Chunk<N, T, M>>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get_mut(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, M>> {
        self.get_mut(key)
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M> {
        self.entry(key).or_insert_with(create_chunk)
    }

    #[inline]
    fn replace(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) -> Option<Chunk<N, T, M>> {
        self.insert(key, chunk)
    }

    #[inline]
    fn write(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) {
        self.insert(key, chunk);
    }
}

impl<'a, N, T, M> IterChunkKeys<'a, N> for FnvHashMap<PointN<N>, Chunk<N, T, M>>
where
    PointN<N>: 'a,
    Chunk<N, T, M>: 'a,
{
    type Iter = hash_map::Keys<'a, PointN<N>, Chunk<N, T, M>>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.keys()
    }
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashMap<N, T, M = ()> = ChunkMap<N, T, M, FnvHashMap<PointN<N>, Chunk<N, T, M>>>;
/// A 2-dimensional `ChunkHashMap`.
pub type ChunkHashMap2<T, M = ()> = ChunkHashMap<[i32; 2], T, M>;
/// A 3-dimensional `ChunkHashMap`.
pub type ChunkHashMap3<T, M = ()> = ChunkHashMap<[i32; 3], T, M>;
