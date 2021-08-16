use crate::{
    chunk::ChunkNode,
    dev_prelude::{ChunkTree, ChunkTreeBuilder, SmallKeyHashMap},
};

use super::{ChunkStorage, IterChunkKeys};

use building_blocks_core::PointN;

use core::hash::Hash;
use std::collections::hash_map;

impl<N, Ch> ChunkStorage<N> for SmallKeyHashMap<PointN<N>, Ch>
where
    PointN<N>: Hash + Eq,
{
    type Chunk = Ch;

    #[inline]
    fn get(&self, key: PointN<N>) -> Option<&Ch> {
        self.get(&key)
    }

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

    #[inline]
    fn delete(&mut self, key: PointN<N>) {
        self.remove(&key);
    }

    #[inline]
    fn pop(&mut self, key: PointN<N>) -> Option<Ch> {
        self.remove(&key)
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

/// A `ChunkTree` using `HashMap` as chunk storage.
pub type HashMapChunkTree<N, T, Bldr> = ChunkTree<
    N,
    T,
    Bldr,
    SmallKeyHashMap<PointN<N>, ChunkNode<<Bldr as ChunkTreeBuilder<N, T>>::Chunk>>,
>;
/// A 2-dimensional `HashMapChunkTree`.
pub type HashMapChunkTree2<T, Bldr> = HashMapChunkTree<[i32; 2], T, Bldr>;
/// A 3-dimensional `HashMapChunkTree`.
pub type HashMapChunkTree3<T, Bldr> = HashMapChunkTree<[i32; 3], T, Bldr>;

pub mod multichannel_aliases {
    use super::*;
    use crate::chunk::tree::multichannel_aliases::*;

    /// An N-dimensional, 1-channel `HashMapChunkTree`.
    pub type HashMapChunkTreeNx1<N, A> = HashMapChunkTree<N, A, ChunkTreeBuilderNx1<N, A>>;

    /// A 2-dimensional, 1-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x1<A> = HashMapChunkTree2<A, ChunkTreeBuilder2x1<A>>;
    /// A 2-dimensional, 2-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x2<A, B> = HashMapChunkTree2<(A, B), ChunkTreeBuilder2x2<A, B>>;
    /// A 2-dimensional, 3-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x3<A, B, C> =
        HashMapChunkTree2<(A, B, C), ChunkTreeBuilder2x3<A, B, C>>;
    /// A 2-dimensional, 4-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x4<A, B, C, D> =
        HashMapChunkTree2<(A, B, C, D), ChunkTreeBuilder2x4<A, B, C, D>>;
    /// A 2-dimensional, 5-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x5<A, B, C, D, E> =
        HashMapChunkTree2<(A, B, C, D, E), ChunkTreeBuilder2x5<A, B, C, D, E>>;
    /// A 2-dimensional, 6-channel `HashMapChunkTree`.
    pub type HashMapChunkTree2x6<A, B, C, D, E, F> =
        HashMapChunkTree2<(A, B, C, D, E, F), ChunkTreeBuilder2x6<A, B, C, D, E, F>>;

    /// A 3-dimensional, 1-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x1<A> = HashMapChunkTree3<A, ChunkTreeBuilder3x1<A>>;
    /// A 3-dimensional, 2-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x2<A, B> = HashMapChunkTree3<(A, B), ChunkTreeBuilder3x2<A, B>>;
    /// A 3-dimensional, 3-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x3<A, B, C> =
        HashMapChunkTree3<(A, B, C), ChunkTreeBuilder3x3<A, B, C>>;
    /// A 3-dimensional, 4-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x4<A, B, C, D> =
        HashMapChunkTree3<(A, B, C, D), ChunkTreeBuilder3x4<A, B, C, D>>;
    /// A 3-dimensional, 5-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x5<A, B, C, D, E> =
        HashMapChunkTree3<(A, B, C, D, E), ChunkTreeBuilder3x5<A, B, C, D, E>>;
    /// A 3-dimensional, 6-channel `HashMapChunkTree`.
    pub type HashMapChunkTree3x6<A, B, C, D, E, F> =
        HashMapChunkTree3<(A, B, C, D, E, F), ChunkTreeBuilder3x6<A, B, C, D, E, F>>;
}

pub use multichannel_aliases::*;
