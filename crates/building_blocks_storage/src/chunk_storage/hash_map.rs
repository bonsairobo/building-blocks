use crate::{ChunkMap, ChunkMapBuilder, SmallKeyHashMap};

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

pub mod multichannel_aliases {
    use super::*;
    use crate::chunk_map::multichannel_aliases::*;

    /// An N-dimensional, 1-channel `ChunkHashMap`.
    pub type ChunkHashMapNx1<N, A> = ChunkHashMap<N, A, ChunkMapBuilderNx1<N, A>>;

    /// A 2-dimensional, 1-channel `ChunkHashMap`.
    pub type ChunkHashMap2x1<A> = ChunkHashMap2<A, ChunkMapBuilder2x1<A>>;
    /// A 2-dimensional, 2-channel `ChunkHashMap`.
    pub type ChunkHashMap2x2<A, B> = ChunkHashMap2<(A, B), ChunkMapBuilder2x2<A, B>>;
    /// A 2-dimensional, 3-channel `ChunkHashMap`.
    pub type ChunkHashMap2x3<A, B, C> = ChunkHashMap2<(A, B, C), ChunkMapBuilder2x3<A, B, C>>;
    /// A 2-dimensional, 4-channel `ChunkHashMap`.
    pub type ChunkHashMap2x4<A, B, C, D> =
        ChunkHashMap2<(A, B, C, D), ChunkMapBuilder2x4<A, B, C, D>>;
    /// A 2-dimensional, 5-channel `ChunkHashMap`.
    pub type ChunkHashMap2x5<A, B, C, D, E> =
        ChunkHashMap2<(A, B, C, D, E), ChunkMapBuilder2x5<A, B, C, D, E>>;
    /// A 2-dimensional, 6-channel `ChunkHashMap`.
    pub type ChunkHashMap2x6<A, B, C, D, E, F> =
        ChunkHashMap2<(A, B, C, D, E, F), ChunkMapBuilder2x6<A, B, C, D, E, F>>;

    /// A 3-dimensional, 1-channel `ChunkHashMap`.
    pub type ChunkHashMap3x1<A> = ChunkHashMap3<A, ChunkMapBuilder3x1<A>>;
    /// A 3-dimensional, 2-channel `ChunkHashMap`.
    pub type ChunkHashMap3x2<A, B> = ChunkHashMap3<(A, B), ChunkMapBuilder3x2<A, B>>;
    /// A 3-dimensional, 3-channel `ChunkHashMap`.
    pub type ChunkHashMap3x3<A, B, C> = ChunkHashMap3<(A, B, C), ChunkMapBuilder3x3<A, B, C>>;
    /// A 3-dimensional, 4-channel `ChunkHashMap`.
    pub type ChunkHashMap3x4<A, B, C, D> =
        ChunkHashMap3<(A, B, C, D), ChunkMapBuilder3x4<A, B, C, D>>;
    /// A 3-dimensional, 5-channel `ChunkHashMap`.
    pub type ChunkHashMap3x5<A, B, C, D, E> =
        ChunkHashMap3<(A, B, C, D, E), ChunkMapBuilder3x5<A, B, C, D, E>>;
    /// A 3-dimensional, 6-channel `ChunkHashMap`.
    pub type ChunkHashMap3x6<A, B, C, D, E, F> =
        ChunkHashMap3<(A, B, C, D, E, F), ChunkMapBuilder3x6<A, B, C, D, E, F>>;
}

pub use multichannel_aliases::*;
