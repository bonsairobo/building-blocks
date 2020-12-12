use crate::{Chunk, ChunkMap, ChunkShape};

use super::ChunkStorage;

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use std::collections::hash_map;

impl<'a, N, T, M> ChunkStorage<'a, N, T, M> for FnvHashMap<PointN<N>, Chunk<N, T, M>>
where
    PointN<N>: Hash + Eq,
    Chunk<N, T, M>: 'a,
{
    #[inline]
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, M>> {
        self.get(key)
    }

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
    fn insert(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) -> Option<Chunk<N, T, M>> {
        self.insert(key, chunk)
    }

    type KeyIter = hash_map::Keys<'a, PointN<N>, Chunk<N, T, M>>;

    #[inline]
    fn iter_keys(&'a self) -> Self::KeyIter {
        self.keys()
    }

    type ChunkIter = hash_map::Iter<'a, PointN<N>, Chunk<N, T, M>>;

    #[inline]
    fn iter_chunks(&'a self) -> Self::ChunkIter {
        self.iter()
    }

    type IntoChunkIter = hash_map::IntoIter<PointN<N>, Chunk<N, T, M>>;

    #[inline]
    fn into_iter_chunks(self) -> Self::IntoChunkIter {
        self.into_iter()
    }
}

/// A `ChunkMap` using `HashMap` as `ChunkStorage`.
pub type ChunkHashMap<N, T> = ChunkMap<N, T, (), FnvHashMap<PointN<N>, Chunk<N, T, ()>>>;
/// A 2-dimensional `ChunkHashMap`.
pub type ChunkHashMap2<T> = ChunkHashMap<[i32; 2], T>;
/// A 3-dimensional `ChunkHashMap`.
pub type ChunkHashMap3<T> = ChunkHashMap<[i32; 3], T>;

impl<'a, N, T, M> ChunkMap<N, T, M, FnvHashMap<PointN<N>, Chunk<N, T, M>>>
where
    PointN<N>: IntegerPoint + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    Chunk<N, T, M>: 'a,
    FnvHashMap<PointN<N>, Chunk<N, T, M>>: ChunkStorage<'a, N, T, M>,
{
    pub fn with_hash_map_storage(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: M,
    ) -> Self {
        Self::new(
            chunk_shape,
            ambient_value,
            default_chunk_metadata,
            FnvHashMap::default(),
        )
    }
}
