use crate::{
    caching::*,
    dev_prelude::{
        ChunkNode, ChunkStorage, ChunkTree, Compressed, Compression, FastArrayCompression,
        FastChannelsCompression, FromBytesCompression, IterChunkKeys, NodeState,
    },
    SmallKeyBuildHasher,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use either::Either;
use slab::Slab;
use thread_local::ThreadLocal;

/// A three-tier chunk storage optimized for memory usage.
///
/// The first tier is an LRU cache of uncompressed chunks. The second tier is a thread-local cache of uncompressed chunks that
/// only gets filled when immutable access requires decompression. The third tier is a `Slab` of compressed chunks.
///
/// Because the thread-local caches are only used when a read misses the main cache, we assume that any data in the main cache
/// is newer than the thread-local cache.
pub struct CompressibleChunkStorage<N, Compr>
where
    N: Send,
    Compr: Compression,
    Compr::Data: Send,
{
    main_cache: SmallKeyLruCache<PointN<N>, ChunkNode<Compr::Data>, CompressedEntry>,
    thread_local_caches: ThreadLocal<LocalChunkCache<N, ChunkNode<Compr::Data>>>,
    compression: Compr,
    compressed: Slab<Compressed<Compr>>,
}

#[derive(Clone)]
pub struct CompressedEntry {
    node_state: NodeState,
    slab_key: usize,
}

type LocalChunkCache<N, Ch> = LocalCache<PointN<N>, Ch, SmallKeyBuildHasher>;

pub type FastCompressibleChunkStorage<N, By, Chan> =
    CompressibleChunkStorage<N, FastArrayCompression<N, FastChannelsCompression<By, Chan>>>;

impl<N, By, Chan> FastCompressibleChunkStorage<N, By, Chan>
where
    N: Send,
    Chan: Send,
    PointN<N>: Hash + IntegerPoint<N>,
    FastChannelsCompression<By, Chan>: Compression<Data = Chan>,
{
    pub fn with_bytes_compression(bytes_compression: By) -> Self {
        Self::new(FastArrayCompression::from_bytes_compression(
            bytes_compression,
        ))
    }
}

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    N: Send,
    Compr: Compression,
    Compr::Data: Send,
{
    pub fn compression(&self) -> &Compr {
        &self.compression
    }
}

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    N: Send,
    PointN<N>: Clone + Eq + Hash,
    Compr: Compression,
    Compr::Data: Send,
{
    pub fn new(compression: Compr) -> Self {
        Self {
            thread_local_caches: Default::default(),
            main_cache: Default::default(),
            compression,
            compressed: Slab::new(),
        }
    }

    pub fn len_cached(&self) -> usize {
        self.main_cache.len_cached()
    }

    pub fn len_compressed(&self) -> usize {
        self.compressed.len()
    }

    pub fn len_total(&self) -> usize {
        self.len_cached() + self.len_compressed()
    }

    pub fn is_empty(&self) -> bool {
        self.len_total() == 0
    }

    // Any time we write new data or delete the data for `key`, we need to invalidate the old locally cached data. Otherwise
    // flushing the local cache would put the cache in an inconsistent state.
    fn invalidate_local_cache_entry(&mut self, key: &PointN<N>) {
        let tlc = self.thread_local_caches.get_or(LocalChunkCache::default);
        unsafe {
            // SAFE: We know that no one is borrowing the data we're deleting, since we have &mut self.
            tlc.delete(key);
        }
    }

    /// Inserts `node` at `key` and returns the old node. Does not require decompression.
    #[inline]
    pub fn insert_node(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Compr::Data>,
    ) -> Option<ChunkNode<Either<Compr::Data, Compressed<Compr>>>> {
        // PERF: we could actually return this entry and avoid decompression
        self.invalidate_local_cache_entry(&key);

        self.main_cache
            .insert(key, node)
            .map(|old_entry| match old_entry {
                CacheEntry::Cached(old_node) => old_node.map(Either::Left),
                CacheEntry::Evicted(compressed_entry) => {
                    let chunk = Either::Right(self.compressed.remove(compressed_entry.slab_key));
                    ChunkNode::new(Some(chunk), compressed_entry.node_state)
                }
            })
    }

    /// Borrow the node at `key`.
    ///
    /// This may trigger decompression of the node's chunk data if it's not cached. In this case, the decompressed data will
    /// live in a thread-local cache until it is either invalidated by newer writes or flushed with `flush_thread_local_caches`.
    #[inline]
    pub fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Compr::Data>> {
        let Self {
            thread_local_caches,
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(value) => value,
            CacheEntry::Evicted(compressed_entry) => thread_local_caches
                .get_or(LocalChunkCache::default)
                .get_or_insert_with(key, || {
                    let chunk = compressed.get(compressed_entry.slab_key).unwrap();
                    ChunkNode::new(
                        Some(chunk.decompress()),
                        compressed_entry.node_state.clone(),
                    )
                }),
        })
    }

    /// Borrow the node state at `key`. Does not require decompression.
    #[inline]
    pub fn get_node_state(&self, key: PointN<N>) -> Option<(&NodeState, bool)> {
        self.main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(node) => (&node.state, node.user_chunk.is_some()),
            CacheEntry::Evicted(compressed_entry) => (&compressed_entry.node_state, true),
        })
    }

    /// Mutably borrow the node at `key`.
    ///
    /// This may trigger decompression of the node's chunk data if it's not cached. In this case, the decompressed data will
    /// live in the main LRU cache until it is compressed again.
    #[inline]
    pub fn get_mut_node(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Compr::Data>> {
        self.invalidate_local_cache_entry(&key);

        let Self {
            main_cache,
            compressed,
            ..
        } = self;

        main_cache.get_mut_or_repopulate_with(key, |compressed_entry| {
            let chunk = compressed.remove(compressed_entry.slab_key);
            ChunkNode::new(Some(chunk.decompress()), compressed_entry.node_state)
        })
    }

    /// Mutably borrow the node state at `key`. Does not require decompression.
    #[inline]
    pub fn get_mut_node_state(&mut self, key: PointN<N>) -> Option<(&mut NodeState, bool)> {
        self.invalidate_local_cache_entry(&key);

        self.main_cache.get_mut(&key).map(|entry| match entry {
            CacheEntry::Cached(node) => (&mut node.state, node.user_chunk.is_some()),
            CacheEntry::Evicted(compressed_entry) => (&mut compressed_entry.node_state, true),
        })
    }

    /// Mutably borrow the node at `key`.
    ///
    /// This may trigger decompression of the node's chunk data if it's not cached. In this case, the decompressed data will
    /// live in the main LRU cache until it is compressed again.
    #[inline]
    pub fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Compr::Data>,
    ) -> &mut ChunkNode<Compr::Data> {
        self.invalidate_local_cache_entry(&key);

        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get_mut_or_insert_with(
            key,
            |compressed_entry| {
                let chunk = compressed.remove(compressed_entry.slab_key);
                ChunkNode::new(
                    Some(chunk.decompress()),
                    compressed_entry.node_state.clone(),
                )
            },
            create_node,
        )
    }

    /// Mutably borrow the node state at `key`. If it doesn't exist, insert the return value of `create_node`. Does not require
    /// decompression.
    #[inline]
    pub fn get_mut_node_state_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Compr::Data>,
    ) -> (&mut NodeState, bool) {
        self.invalidate_local_cache_entry(&key);

        let Self { main_cache, .. } = self;

        let entry = main_cache.get_mut_or_insert_without_repopulate(key, create_node);

        match entry {
            CacheEntry::Cached(node) => (&mut node.state, node.user_chunk.is_some()),
            CacheEntry::Evicted(compressed_entry) => (&mut compressed_entry.node_state, true),
        }
    }

    /// Remove the node at `key`.
    ///
    /// This may trigger decompression of the node's chunk data if it's not cached.
    #[inline]
    pub fn pop_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Compr::Data>> {
        self.invalidate_local_cache_entry(&key);

        self.main_cache.remove(&key).map(|entry| match entry {
            CacheEntry::Cached(node) => node,
            CacheEntry::Evicted(compressed_entry) => {
                let chunk = self
                    .compressed
                    .remove(compressed_entry.slab_key)
                    .decompress();
                ChunkNode::new(Some(chunk), compressed_entry.node_state)
            }
        })
    }

    /// Remove the node at `key`. Does not require decompression.
    #[inline]
    pub fn pop_raw_node(
        &mut self,
        key: PointN<N>,
    ) -> Option<ChunkNode<Either<Compr::Data, Compressed<Compr>>>> {
        self.invalidate_local_cache_entry(&key);

        self.main_cache.remove(&key).map(|entry| match entry {
            CacheEntry::Cached(node) => node.map(Either::Left),
            CacheEntry::Evicted(compressed_entry) => {
                let compressed_chunk = self.compressed.remove(compressed_entry.slab_key);
                ChunkNode::new(
                    Some(Either::Right(compressed_chunk)),
                    compressed_entry.node_state,
                )
            }
        })
    }

    /// Writes `chunk` into the node at `key`, leaving any other state unaffected. Does not require decompression.
    ///
    /// The node's state and a `bool` indicating whether any old data was overwritten are returned for convenience.
    #[inline]
    pub fn write_chunk(&mut self, key: PointN<N>, chunk: Compr::Data) -> (&mut NodeState, bool) {
        self.invalidate_local_cache_entry(&key);

        let Self {
            main_cache,
            compressed,
            ..
        } = self;

        let node = main_cache.get_mut_or_insert_with(
            key,
            |compressed_entry| {
                compressed.remove(compressed_entry.slab_key);
                ChunkNode::new_without_data(compressed_entry.node_state.clone())
            },
            ChunkNode::new_empty,
        );
        let had_data = node.user_chunk.is_some();
        node.user_chunk = Some(chunk);

        (&mut node.state, had_data)
    }

    /// Deletes the chunk out of the node at `key`, leaving any other state unaffected. Does not require decompression.
    ///
    /// The node's state is returned for convenience.
    #[inline]
    pub fn delete_chunk(&mut self, key: PointN<N>) -> Option<NodeState> {
        self.invalidate_local_cache_entry(&key);

        let Self {
            main_cache,
            compressed,
            ..
        } = self;

        if let Some(entry) = main_cache.get_mut(&key) {
            match entry {
                CacheEntry::Cached(node) => {
                    node.user_chunk = None;

                    Some(node.state.clone())
                }
                CacheEntry::Evicted(compressed_entry) => {
                    compressed.remove(compressed_entry.slab_key);
                    // Promote to the LRU cache with just the node state.
                    let node_state = compressed_entry.node_state.clone();
                    let node = ChunkNode::new_without_data(node_state.clone());
                    main_cache.insert(key, node);

                    Some(node_state)
                }
            }
        } else {
            None
        }
    }

    /// Tries to fetch the node from three different tiers in order:
    ///
    /// 1. main cache
    /// 2. thread local cache
    /// 3. compressed storage
    ///
    /// WARNING: the cache will not be updated.
    #[inline]
    pub fn get_node_without_caching(
        &self,
        key: PointN<N>,
    ) -> Option<(&NodeState, Either<Option<&Compr::Data>, &Compressed<Compr>>)> {
        let Self {
            thread_local_caches,
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(node) => (&node.state, Either::Left(node.user_chunk.as_ref())),
            CacheEntry::Evicted(compressed_entry) => {
                let chunk = thread_local_caches
                    .get_or(LocalChunkCache::default)
                    .get(key)
                    .map(|node| Either::Left(node.user_chunk.as_ref()))
                    .unwrap_or_else(|| {
                        Either::Right(compressed.get(compressed_entry.slab_key).unwrap())
                    });
                (&compressed_entry.node_state, chunk)
            }
        })
    }

    /// Compress the least-recently-used, cached entry. Returns `true` iff a chunk was compressed.
    ///
    /// There may not be any data to compress in the LRU node, in which case nothing will be compressed, and the node will be
    /// re-inserted into the cache (becoming most-recently used).
    #[inline]
    pub fn try_compress_lru(&mut self) -> bool {
        let CompressibleChunkStorage {
            main_cache,
            compression,
            ..
        } = self;

        if let Some((key, node)) = main_cache.remove_lru() {
            if let Some(chunk) = node.user_chunk {
                let compressed_chunk = compression.compress(&chunk);
                self._insert_compressed(key, node.state, compressed_chunk);
                true
            } else {
                main_cache.insert(key, node);
                false
            }
        } else {
            false
        }
    }

    /// Remove the least-recently-used, cached chunk.
    #[inline]
    pub fn remove_lru(&mut self) -> Option<(PointN<N>, ChunkNode<Compr::Data>)> {
        self.main_cache.remove_lru().map(|(k, v)| {
            self.invalidate_local_cache_entry(&k);
            (k, v)
        })
    }

    /// Insert a node with a compressed chunk. Returns the old node if one exists.
    #[inline]
    pub fn insert_compressed(
        &mut self,
        key: PointN<N>,
        node_state: NodeState,
        chunk: Compressed<Compr>,
    ) -> Option<ChunkNode<Either<Compr::Data, Compressed<Compr>>>> {
        self.invalidate_local_cache_entry(&key);
        self._insert_compressed(key, node_state, chunk)
    }

    fn _insert_compressed(
        &mut self,
        key: PointN<N>,
        node_state: NodeState,
        chunk: Compressed<Compr>,
    ) -> Option<ChunkNode<Either<Compr::Data, Compressed<Compr>>>> {
        self.invalidate_local_cache_entry(&key);

        let slab_key = self.compressed.insert(chunk);
        let old_entry = self.main_cache.evict(
            key,
            CompressedEntry {
                slab_key,
                node_state,
            },
        );

        old_entry.map(|entry| match entry {
            CacheEntry::Cached(node) => node.map(Either::Left),
            CacheEntry::Evicted(compressed_entry) => {
                let chunk = self.compressed.remove(compressed_entry.slab_key);
                ChunkNode::new(Some(Either::Right(chunk)), compressed_entry.node_state)
            }
        })
    }

    /// Consumes and flushes all thread local caches into the global cache. This should be done occasionally to reduce memory
    /// usage and improve caching efficiency.
    #[inline]
    pub fn flush_thread_local_caches(&mut self) {
        let taken_caches = std::mem::replace(&mut self.thread_local_caches, ThreadLocal::new());
        for mut cache in taken_caches.into_iter() {
            for (k, v) in cache.drain_iter() {
                // This only keeps the locally cached value if the main cache entry is evicted. This prevents us from
                // overwriting newer data.
                self.main_cache.get_mut_or_repopulate_with(k, |_| v);
            }
        }
    }
}

impl<N, Compr> ChunkStorage<N> for CompressibleChunkStorage<N, Compr>
where
    N: Send,
    PointN<N>: Clone + Eq + Hash,
    Compr: Compression,
    Compr::Data: 'static + Send,
{
    type Chunk = Compr::Data;
    type ColdChunk = Compressed<Compr>;

    #[inline]
    fn insert_node(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Either<Self::Chunk, Self::ColdChunk>>> {
        self.insert_node(key, node)
    }

    #[inline]
    fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>> {
        self.get_node(key)
    }

    #[inline]
    fn get_raw_node(
        &self,
        key: PointN<N>,
    ) -> Option<(&NodeState, Either<Option<&Self::Chunk>, &Self::ColdChunk>)> {
        self.get_node_without_caching(key)
    }

    #[inline]
    fn get_node_state(&self, key: PointN<N>) -> Option<(&NodeState, bool)> {
        self.get_node_state(key)
    }

    #[inline]
    fn get_mut_node(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Self::Chunk>> {
        self.get_mut_node(key)
    }

    #[inline]
    fn get_mut_node_state(&mut self, key: PointN<N>) -> Option<(&mut NodeState, bool)> {
        self.get_mut_node_state(key)
    }

    #[inline]
    fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut ChunkNode<Self::Chunk> {
        self.get_mut_node_or_insert_with(key, create_node)
    }

    #[inline]
    fn get_mut_node_state_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> (&mut NodeState, bool) {
        self.get_mut_node_state_or_insert_with(key, create_node)
    }

    #[inline]
    fn pop_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::Chunk>> {
        self.pop_node(key)
    }

    #[inline]
    fn pop_raw_node(
        &mut self,
        key: PointN<N>,
    ) -> Option<ChunkNode<Either<Self::Chunk, Self::ColdChunk>>> {
        self.pop_raw_node(key)
    }

    #[inline]
    fn write_chunk(&mut self, key: PointN<N>, chunk: Self::Chunk) -> (&mut NodeState, bool) {
        self.write_chunk(key, chunk)
    }

    #[inline]
    fn delete_chunk(&mut self, key: PointN<N>) -> Option<NodeState> {
        self.delete_chunk(key)
    }
}

impl<'a, N: 'a, Compr> IterChunkKeys<'a, N> for CompressibleChunkStorage<N, Compr>
where
    N: Send,
    PointN<N>: Clone + Eq + Hash,
    Compr: Compression,
    Compr::Data: 'a + Send,
{
    type Iter = LruChunkCacheKeys<'a, N, Compr::Data>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.main_cache.keys()
    }
}

impl<'a, N, Compr> IntoIterator for &'a CompressibleChunkStorage<N, Compr>
where
    N: Send,
    Compr: Compression,
    Compr::Data: Send,
    PointN<N>: Clone + Eq + Hash,
{
    type IntoIter = Box<dyn 'a + Iterator<Item = Self::Item>>;
    type Item = (&'a PointN<N>, &'a ChunkNode<Compr::Data>);

    fn into_iter(self) -> Self::IntoIter {
        let CompressibleChunkStorage {
            main_cache,
            thread_local_caches,
            compressed,
            ..
        } = self;
        Box::new(main_cache.entries().map(move |(key, entry)| match entry {
            CacheEntry::Cached(chunk) => (key, chunk),
            CacheEntry::Evicted(compressed_entry) => {
                let local_cache = thread_local_caches.get_or(LocalChunkCache::default);
                let chunk = local_cache.get_or_insert_with(key.clone(), || {
                    let chunk = compressed.get(compressed_entry.slab_key).unwrap();
                    ChunkNode::new(
                        Some(chunk.decompress()),
                        compressed_entry.node_state.clone(),
                    )
                });
                (key, chunk)
            }
        }))
    }
}

impl<N: 'static, Compr: 'static> IntoIterator for CompressibleChunkStorage<N, Compr>
where
    N: Send,
    Compr: Compression,
    Compr::Data: Send,
{
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;
    type Item = (PointN<N>, ChunkNode<Compr::Data>);

    fn into_iter(self) -> Self::IntoIter {
        let Self {
            main_cache,
            mut compressed,
            ..
        } = self;
        Box::new(main_cache.into_iter().map(move |(key, entry)| match entry {
            CacheEntry::Cached(chunk) => (key, chunk),
            CacheEntry::Evicted(compressed_entry) => (
                key,
                ChunkNode::new(
                    Some(compressed.remove(compressed_entry.slab_key).decompress()),
                    compressed_entry.node_state,
                ),
            ),
        }))
    }
}

pub type SlabKey = usize;

pub type LruChunkCacheKeys<'a, N, Ch> = LruCacheKeys<'a, PointN<N>, ChunkNode<Ch>, CompressedEntry>;
pub type LruChunkCacheEntries<'a, N, Ch> =
    LruCacheEntries<'a, PointN<N>, ChunkNode<Ch>, CompressedEntry>;
pub type LruChunkCacheIntoIter<N, Ch> = LruCacheIntoIter<PointN<N>, ChunkNode<Ch>, CompressedEntry>;

/// A `ChunkTree` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkTree<N, T, Bldr, Compr> =
    ChunkTree<N, T, Bldr, CompressibleChunkStorage<N, Compr>>;

pub mod multichannel_aliases {
    use super::*;
    use crate::array::compression::multichannel_aliases::*;
    use crate::dev_prelude::{Channel, ChunkTreeBuilderNxM};

    pub type FastCompressibleChunkStorageNx1<N, By, A> =
        CompressibleChunkStorage<N, FastArrayCompressionNx1<N, By, A>>;
    pub type FastCompressibleChunkStorageNx2<N, By, A, B> =
        CompressibleChunkStorage<N, FastArrayCompressionNx2<N, By, A, B>>;
    pub type FastCompressibleChunkStorageNx3<N, By, A, B, C> =
        CompressibleChunkStorage<N, FastArrayCompressionNx3<N, By, A, B, C>>;
    pub type FastCompressibleChunkStorageNx4<N, By, A, B, C, D> =
        CompressibleChunkStorage<N, FastArrayCompressionNx4<N, By, A, B, C, D>>;
    pub type FastCompressibleChunkStorageNx5<N, By, A, B, C, D, E> =
        CompressibleChunkStorage<N, FastArrayCompressionNx5<N, By, A, B, C, D, E>>;
    pub type FastCompressibleChunkStorageNx6<N, By, A, B, C, D, E, F> =
        CompressibleChunkStorage<N, FastArrayCompressionNx6<N, By, A, B, C, D, E, F>>;

    macro_rules! compressible_map_type_alias {
        ($name:ident, $dim:ty, $( $chan:ident ),+ ) => {
            pub type $name<By, $( $chan ),+> = CompressibleChunkTree<
                $dim,
                ($($chan,)+),
                ChunkTreeBuilderNxM<$dim, ($($chan,)+), ($(Channel<$chan>,)+)>,
                FastArrayCompression<$dim, FastChannelsCompression<By, ($(Channel<$chan>,)+)>>,
            >;
        };
    }

    pub type CompressibleChunkTreeNx1<N, By, A> = CompressibleChunkTree<
        N,
        A,
        ChunkTreeBuilderNxM<N, A, Channel<A>>,
        FastArrayCompression<N, FastChannelsCompression<By, Channel<A>>>,
    >;

    pub type CompressibleChunkTree2x1<By, A> = CompressibleChunkTreeNx1<[i32; 2], By, A>;
    compressible_map_type_alias!(CompressibleChunkTree2x2, [i32; 2], A, B);
    compressible_map_type_alias!(CompressibleChunkTree2x3, [i32; 2], A, B, C);
    compressible_map_type_alias!(CompressibleChunkTree2x4, [i32; 2], A, B, C, D);
    compressible_map_type_alias!(CompressibleChunkTree2x5, [i32; 2], A, B, C, D, E);
    compressible_map_type_alias!(CompressibleChunkTree2x6, [i32; 2], A, B, C, D, E, F);

    pub type CompressibleChunkTree3x1<By, A> = CompressibleChunkTreeNx1<[i32; 3], By, A>;
    compressible_map_type_alias!(CompressibleChunkTree3x2, [i32; 3], A, B);
    compressible_map_type_alias!(CompressibleChunkTree3x3, [i32; 3], A, B, C);
    compressible_map_type_alias!(CompressibleChunkTree3x4, [i32; 3], A, B, C, D);
    compressible_map_type_alias!(CompressibleChunkTree3x5, [i32; 3], A, B, C, D, E);
    compressible_map_type_alias!(CompressibleChunkTree3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;
