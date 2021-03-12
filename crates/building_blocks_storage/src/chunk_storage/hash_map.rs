use crate::{ChunkMap, ChunkMapBuilder, ChunkMapBuilderNx1, SmallKeyHashMap};

use super::{ChunkReadStorage, ChunkWriteStorage, IterChunkKeys};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use std::collections::hash_map;

impl<N, Ch> ChunkReadStorage<N, Ch> for SmallKeyHashMap<PointN<N>, Ch>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get(&self, key: PointN<N>) -> Option<&Ch> {
        self.get(&key)
    }
}

impl<N, Ch> ChunkWriteStorage<N, Ch> for SmallKeyHashMap<PointN<N>, Ch>
where
    PointN<N>: Hash + Eq,
{
    #[inline]
    fn get_mut(&mut self, key: PointN<N>) -> Option<&mut Ch> {
        self.get_mut(&key)
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Ch,
    ) -> &mut Ch {
        self.entry(key).or_insert_with(create_chunk)
    }

    #[inline]
    fn replace(&mut self, key: PointN<N>, chunk: Ch) -> Option<Ch> {
        self.insert(key, chunk)
    }

    #[inline]
    fn write(&mut self, key: PointN<N>, chunk: Ch) {
        self.insert(key, chunk);
    }
}

impl<'a, N, Ch> IterChunkKeys<'a, N> for SmallKeyHashMap<PointN<N>, Ch>
where
    PointN<N>: 'a,
    Ch: 'a,
{
    type Iter = hash_map::Keys<'a, PointN<N>, Ch>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.keys()
    }
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashMap<N, T, B> =
    ChunkMap<N, T, B, SmallKeyHashMap<PointN<N>, <B as ChunkMapBuilder<N, T>>::Chunk>>;
/// A 2-dimensional `ChunkHashMap`.
pub type ChunkHashMap2<T, B> = ChunkHashMap<[i32; 2], T, B>;
/// A 3-dimensional `ChunkHashMap`.
pub type ChunkHashMap3<T, B> = ChunkHashMap<[i32; 3], T, B>;

/// An N-dimensional, 1-channel `ChunkHashMap`.
pub type ChunkHashMapNx1<N, T> = ChunkHashMap<N, T, ChunkMapBuilderNx1<N, T>>;
/// A 2-dimensional, 1-channel `ChunkHashMap`.
pub type ChunkHashMap2x1<T> = ChunkHashMap<[i32; 2], T, ChunkMapBuilderNx1<[i32; 2], T>>;
/// A 3-dimensional, 1-channel `ChunkHashMap`.
pub type ChunkHashMap3x1<T> = ChunkHashMap<[i32; 3], T, ChunkMapBuilderNx1<[i32; 3], T>>;
