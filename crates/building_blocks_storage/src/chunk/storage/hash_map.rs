use crate::{
    chunk::ChunkNode,
    dev_prelude::{ChunkMap, ChunkMapBuilder, SmallKeyHashMap},
};

use super::{ChunkKey, ChunkStorage, IterChunkKeys};

use core::hash::Hash;
use std::collections::hash_map;

impl<N, Ch> ChunkStorage<N> for SmallKeyHashMap<ChunkKey<N>, Ch>
where
    ChunkKey<N>: Hash + Eq,
{
    type Chunk = Ch;

    #[inline]
    fn get(&self, key: ChunkKey<N>) -> Option<&Ch> {
        self.get(&key)
    }

    #[inline]
    fn get_mut(&mut self, key: ChunkKey<N>) -> Option<&mut Ch> {
        self.get_mut(&key)
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
        create_chunk: impl FnOnce() -> Ch,
    ) -> &mut Ch {
        self.entry(key).or_insert_with(create_chunk)
    }

    #[inline]
    fn replace(&mut self, key: ChunkKey<N>, chunk: Ch) -> Option<Ch> {
        self.insert(key, chunk)
    }

    #[inline]
    fn write(&mut self, key: ChunkKey<N>, chunk: Ch) {
        self.insert(key, chunk);
    }

    #[inline]
    fn delete(&mut self, key: ChunkKey<N>) {
        self.remove(&key);
    }

    #[inline]
    fn pop(&mut self, key: ChunkKey<N>) -> Option<Ch> {
        self.remove(&key)
    }
}

impl<'a, N, Ch> IterChunkKeys<'a, N> for SmallKeyHashMap<ChunkKey<N>, Ch>
where
    ChunkKey<N>: 'a,
    Ch: 'a,
{
    type Iter = hash_map::Keys<'a, ChunkKey<N>, Ch>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.keys()
    }
}

/// A `ChunkMap` using `HashMap` as chunk storage.
pub type ChunkHashMap<N, T, Bldr> = ChunkMap<
    N,
    T,
    Bldr,
    SmallKeyHashMap<ChunkKey<N>, ChunkNode<<Bldr as ChunkMapBuilder<N, T>>::Chunk>>,
>;
/// A 2-dimensional `ChunkHashMap`.
pub type ChunkHashMap2<T, Bldr> = ChunkHashMap<[i32; 2], T, Bldr>;
/// A 3-dimensional `ChunkHashMap`.
pub type ChunkHashMap3<T, Bldr> = ChunkHashMap<[i32; 3], T, Bldr>;

pub mod multichannel_aliases {
    use super::*;
    use crate::chunk::map::multichannel_aliases::*;

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
