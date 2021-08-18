use crate::dev_prelude::{ChunkTree, ChunkTreeBuilder, SmallKeyHashMap};

use super::{ChunkNode, ChunkStorage, IterChunkKeys};

use building_blocks_core::PointN;

use core::hash::Hash;
use std::collections::hash_map;

impl<N, Ch> ChunkStorage<N> for SmallKeyHashMap<PointN<N>, ChunkNode<Ch>>
where
    PointN<N>: Hash + Eq,
{
    type Chunk = Ch;
    type ChunkRepr = Ch;

    #[inline]
    fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>> {
        self.get(&key)
    }

    #[inline]
    fn get_mut_node(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Self::Chunk>> {
        self.get_mut(&key)
    }

    #[inline]
    fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut ChunkNode<Self::Chunk> {
        self.entry(key).or_insert_with(create_node)
    }

    #[inline]
    fn replace_node(
        &mut self,
        key: PointN<N>,
        chunk: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Self::Chunk>> {
        self.insert(key, chunk)
    }

    #[inline]
    fn write_node(&mut self, key: PointN<N>, chunk: ChunkNode<Self::Chunk>) {
        self.insert(key, chunk);
    }

    #[inline]
    fn write_raw_node(&mut self, key: PointN<N>, chunk: ChunkNode<Self::Chunk>) {
        self.insert(key, chunk);
    }

    #[inline]
    fn pop_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::Chunk>> {
        self.remove(&key)
    }

    #[inline]
    fn pop_raw_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::ChunkRepr>> {
        self.remove(&key)
    }

    #[inline]
    fn write_chunk(&mut self, key: PointN<N>, chunk: Self::Chunk) {
        let node = self.entry(key).or_insert_with(ChunkNode::new_empty);
        node.user_chunk = Some(chunk);
    }

    #[inline]
    fn get_child_mask(&self, key: PointN<N>) -> Option<u8> {
        self.get(&key).map(|n| n.child_mask)
    }

    #[inline]
    fn get_mut_child_mask(&mut self, key: PointN<N>) -> Option<(&mut u8, bool)> {
        self.get_mut(&key)
            .map(|n| (&mut n.child_mask, n.user_chunk.is_some()))
    }

    #[inline]
    fn get_mut_child_mask_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut u8 {
        let node = self.entry(key).or_insert_with(create_node);
        &mut node.child_mask
    }
}

impl<'a, N, Ch> IterChunkKeys<'a, N> for SmallKeyHashMap<PointN<N>, ChunkNode<Ch>>
where
    PointN<N>: 'a,
    Ch: 'a,
{
    type Iter = hash_map::Keys<'a, PointN<N>, ChunkNode<Ch>>;

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
    use crate::chunk::tree::builder::multichannel_aliases::*;

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
