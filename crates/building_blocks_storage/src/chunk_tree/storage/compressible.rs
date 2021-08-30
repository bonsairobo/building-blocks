use crate::{
    caching::*,
    compression::MaybeCompressed,
    dev_prelude::{
        ChunkNode, ChunkStorage, ChunkTree, Compressed, Compression, FastArrayCompression,
        FastChannelsCompression, FromBytesCompression, IterChunkKeys, NodeState,
    },
    SmallKeyBuildHasher,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use slab::Slab;
use thread_local::ThreadLocal;

/// A three-tier chunk storage optimized for memory usage.
///
/// The first tier is an LRU cache of uncompressed chunks. The second tier is a thread-local cache of uncompressed chunks that
/// only gets written to when immutable access requires decompression. The third tier is a `Slab` of compressed chunks.
pub struct CompressibleChunkStorage<N, Compr>
where
    N: Send,
    Compr: Compression,
    Compr::Data: Send,
{
    main_cache: SmallKeyLruCache<PointN<N>, ChunkNode<Compr::Data>, CompressedLocation>,
    thread_local_caches: ThreadLocal<LocalChunkCache<N, ChunkNode<Compr::Data>>>,
    compression: Compr,
    compressed: CompressedChunks<Compr>,
}

type CompressedChunks<Compr> = Slab<ChunkNode<Compressed<Compr>>>;

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

    /// Tries to fetch the value from three different tiers in order:
    ///
    /// 1. main cache
    /// 2. thread local cache
    /// 3. compressed storage
    ///
    /// WARNING: the cache will not be updated.
    pub fn get_without_caching(
        &self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<&ChunkNode<Compr::Data>, &ChunkNode<Compressed<Compr>>>> {
        let Self {
            thread_local_caches,
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(value) => MaybeCompressed::Decompressed(value),
            CacheEntry::Evicted(location) => thread_local_caches
                .get_or(LocalChunkCache::default)
                .get(key)
                .map(MaybeCompressed::Decompressed)
                .unwrap_or_else(|| {
                    MaybeCompressed::Compressed(compressed.get(location.0).unwrap())
                }),
        })
    }

    /// Same as `get_without_caching`, but it skips over the thread-local cache. This only has the effect of eliding one hash
    /// map lookup if we don't need to prioritize finding a decompressed chunk.
    ///
    /// WARNING: the cache will not be updated.
    pub fn get_without_caching_and_skip_thread_local(
        &self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<&ChunkNode<Compr::Data>, &ChunkNode<Compressed<Compr>>>> {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(value) => MaybeCompressed::Decompressed(value),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(compressed.get(location.0).unwrap())
            }
        })
    }

    /// Same as `get_without_caching`, but it skips over the thread-local cache. This only has the effect of eliding one hash
    /// map lookup if we don't need to prioritize finding a decompressed chunk.
    ///
    /// WARNING: the cache will not be updated.
    pub fn get_mut_without_caching_and_skip_thread_local(
        &mut self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<&mut ChunkNode<Compr::Data>, &mut ChunkNode<Compressed<Compr>>>>
    {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get_mut(&key).map(move |entry| match entry {
            CacheEntry::Cached(value) => MaybeCompressed::Decompressed(value),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(compressed.get_mut(location.0).unwrap())
            }
        })
    }

    /// Same as `get_mut_without_caching_and_skip_thread_local`, but it will fill the entry with `create_node`.
    ///
    /// WARNING: the cache will not be updated.
    pub fn get_mut_or_insert_without_caching_and_skip_thread_local(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Compr::Data>,
    ) -> MaybeCompressed<&mut ChunkNode<Compr::Data>, &mut ChunkNode<Compressed<Compr>>> {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        match main_cache.get_mut_or_insert_without_repopulate(key, create_node) {
            CacheEntry::Cached(value) => MaybeCompressed::Decompressed(value),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(compressed.get_mut(location.0).unwrap())
            }
        }
    }

    /// Returns a clone of the entry at `key`.
    ///
    /// WARNING: the cache will not be updated. This method should be used for a read-modify-write workflow where it would be
    /// inefficient to cache the chunk only for it to be overwritten by the modified version.
    pub fn clone_without_caching(
        &self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<ChunkNode<Compr::Data>, ChunkNode<Compressed<Compr>>>>
    where
        Compr::Data: Clone,
        Compressed<Compr>: Clone,
    {
        self.get_without_caching(key).map(|e| match e {
            MaybeCompressed::Compressed(c) => MaybeCompressed::Compressed(c.clone()),
            MaybeCompressed::Decompressed(d) => MaybeCompressed::Decompressed(d.clone()),
        })
    }

    /// Remove the entry at `key`.
    pub fn remove(
        &mut self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<ChunkNode<Compr::Data>, ChunkNode<Compressed<Compr>>>> {
        self.main_cache.remove(&key).map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressed::Decompressed(chunk),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(self.compressed.remove(location.0))
            }
        })
    }

    /// Compress the least-recently-used, cached entry.
    pub fn compress_lru(&mut self) {
        let CompressibleChunkStorage {
            main_cache,
            compressed,
            compression,
            ..
        } = self;
        let compressed_entry = compressed.vacant_entry();
        if let Some((_, node)) = main_cache.evict_lru(CompressedLocation(compressed_entry.key())) {
            compressed_entry.insert(node.as_ref().map(|c| compression.compress(c)));
        }
    }

    /// Remove the least-recently-used, cached chunk.
    pub fn remove_lru(&mut self) -> Option<(PointN<N>, ChunkNode<Compr::Data>)> {
        self.main_cache.remove_lru()
    }

    /// Insert a node with a compressed chunk. Returns the old node if one exists.
    pub fn insert_compressed(
        &mut self,
        key: PointN<N>,
        compressed_chunk: ChunkNode<Compressed<Compr>>,
    ) -> Option<MaybeCompressed<ChunkNode<Compr::Data>, ChunkNode<Compressed<Compr>>>> {
        let compressed_entry = self.compressed.vacant_entry();
        let old_entry = self
            .main_cache
            .evict(key, CompressedLocation(compressed_entry.key()));
        compressed_entry.insert(compressed_chunk);

        old_entry.map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressed::Decompressed(chunk),
            CacheEntry::Evicted(old_location) => {
                MaybeCompressed::Compressed(self.compressed.remove(old_location.0))
            }
        })
    }

    /// Consumes and flushes all thread local caches into the global cache. This should be done occasionally to reduce memory
    /// usage and improve caching efficiency.
    pub fn flush_thread_local_caches(&mut self) {
        let taken_caches = std::mem::replace(&mut self.thread_local_caches, ThreadLocal::new());
        for mut cache in taken_caches.into_iter() {
            for (k, v) in cache.drain_iter() {
                self.main_cache.insert(k, v);
            }
        }
    }

    /// Inserts `node` at `key` and returns the old node.
    pub fn insert(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Compr::Data>,
    ) -> Option<MaybeCompressed<ChunkNode<Compr::Data>, ChunkNode<Compressed<Compr>>>> {
        self.main_cache
            .insert(key, node)
            .map(|old_entry| match old_entry {
                CacheEntry::Cached(old_chunk) => MaybeCompressed::Decompressed(old_chunk),
                CacheEntry::Evicted(location) => {
                    MaybeCompressed::Compressed(self.compressed.remove(location.0))
                }
            })
    }
}

impl<N, Compr> ChunkStorage<N> for CompressibleChunkStorage<N, Compr>
where
    N: Send,
    PointN<N>: Clone + Eq + Hash,
    Compr: Compression,
    Compr::Data: Send,
{
    type Chunk = Compr::Data;
    type ChunkRepr = MaybeCompressed<Self::Chunk, Compressed<Compr>>;

    #[inline]
    fn contains_chunk(&self, key: PointN<N>) -> bool {
        self.get_without_caching_and_skip_thread_local(key)
            .map(|e| match e {
                MaybeCompressed::Compressed(n) => n.user_chunk.is_some(),
                MaybeCompressed::Decompressed(n) => n.user_chunk.is_some(),
            })
            .unwrap_or(false)
    }

    /// Borrow the node at `key`.
    fn get_node(&self, key: PointN<N>) -> Option<&ChunkNode<Self::Chunk>> {
        let Self {
            thread_local_caches,
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(value) => value,
            CacheEntry::Evicted(location) => thread_local_caches
                .get_or(LocalChunkCache::default)
                .get_or_insert_with(key, || {
                    compressed
                        .get(location.0)
                        .unwrap()
                        .as_ref()
                        .map(Compressed::decompress)
                }),
        })
    }

    #[inline]
    fn get_mut_node(&mut self, key: PointN<N>) -> Option<&mut ChunkNode<Self::Chunk>> {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;

        main_cache.get_mut_or_repopulate_with(key, |location| {
            compressed
                .remove(location.0)
                .as_ref()
                .map(Compressed::decompress)
        })
    }

    #[inline]
    fn get_mut_node_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_node: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut ChunkNode<Self::Chunk> {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        main_cache.get_mut_or_insert_with(
            key,
            |location| {
                compressed
                    .remove(location.0)
                    .as_ref()
                    .map(Compressed::decompress)
            },
            create_node,
        )
    }

    #[inline]
    fn replace_node(
        &mut self,
        key: PointN<N>,
        node: ChunkNode<Self::Chunk>,
    ) -> Option<ChunkNode<Self::Chunk>> {
        self.insert(key, node).map(|old_chunk| match old_chunk {
            MaybeCompressed::Decompressed(old_chunk) => old_chunk,
            MaybeCompressed::Compressed(old_chunk) => {
                old_chunk.as_ref().map(Compressed::decompress)
            }
        })
    }

    #[inline]
    fn write_node(&mut self, key: PointN<N>, chunk: ChunkNode<Self::Chunk>) {
        self.insert(key, chunk);
    }

    #[inline]
    fn write_raw_node(&mut self, key: PointN<N>, node: ChunkNode<Self::ChunkRepr>) {
        if let Some(user_chunk) = node.user_chunk {
            match user_chunk {
                MaybeCompressed::Compressed(c) => {
                    self.insert_compressed(key, ChunkNode::new(Some(c), node.state));
                }
                MaybeCompressed::Decompressed(d) => {
                    self.write_node(key, ChunkNode::new(Some(d), node.state))
                }
            }
        } else {
            self.write_node(key, ChunkNode::new_without_data(node.state))
        }
    }

    #[inline]
    fn pop_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::Chunk>> {
        self.remove(key)
            .map(|ch| ch.into_decompressed_with(|c| c.as_ref().map(Compressed::decompress)))
    }

    #[inline]
    fn pop_raw_node(&mut self, key: PointN<N>) -> Option<ChunkNode<Self::ChunkRepr>> {
        self.remove(key).map(|c| match c {
            MaybeCompressed::Compressed(n) => n.map(MaybeCompressed::Compressed),
            MaybeCompressed::Decompressed(n) => n.map(MaybeCompressed::Decompressed),
        })
    }

    #[inline]
    fn write_chunk(&mut self, key: PointN<N>, chunk: Self::Chunk) {
        let Self {
            main_cache,
            compressed,
            ..
        } = self;
        let node = main_cache.get_mut_or_insert_with(
            key,
            |location| {
                let node = compressed.remove(location.0);
                ChunkNode::new_without_data(node.state)
            },
            ChunkNode::new_empty,
        );
        node.user_chunk = Some(chunk);
    }

    #[inline]
    fn get_node_state(&self, key: PointN<N>) -> Option<&NodeState> {
        self.get_without_caching_and_skip_thread_local(key)
            .map(|e| match e {
                MaybeCompressed::Compressed(n) => &n.state,
                MaybeCompressed::Decompressed(n) => &n.state,
            })
    }

    #[inline]
    fn get_mut_node_state(&mut self, key: PointN<N>) -> Option<(&mut NodeState, bool)> {
        self.get_mut_without_caching_and_skip_thread_local(key)
            .map(|e| match e {
                MaybeCompressed::Compressed(n) => (&mut n.state, n.user_chunk.is_some()),
                MaybeCompressed::Decompressed(n) => (&mut n.state, n.user_chunk.is_some()),
            })
    }

    #[inline]
    fn get_mut_node_state_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> ChunkNode<Self::Chunk>,
    ) -> &mut NodeState {
        match self.get_mut_or_insert_without_caching_and_skip_thread_local(key, create_chunk) {
            MaybeCompressed::Compressed(n) => &mut n.state,
            MaybeCompressed::Decompressed(n) => &mut n.state,
        }
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
            CacheEntry::Evicted(location) => {
                let local_cache = thread_local_caches.get_or(LocalChunkCache::default);
                let chunk = local_cache.get_or_insert_with(key.clone(), || {
                    compressed
                        .get(location.0)
                        .unwrap()
                        .as_ref()
                        .map(Compressed::decompress)
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
        Box::new(main_cache.into_iter().map(move |(key, entry)| {
            match entry {
                CacheEntry::Cached(chunk) => (key, chunk),
                CacheEntry::Evicted(location) => (
                    key,
                    compressed
                        .remove(location.0)
                        .as_ref()
                        .map(Compressed::decompress),
                ),
            }
        }))
    }
}

/// An index into a compressed chunk slab.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedLocation(pub usize);

pub type LruChunkCacheKeys<'a, N, Ch> =
    LruCacheKeys<'a, PointN<N>, ChunkNode<Ch>, CompressedLocation>;
pub type LruChunkCacheEntries<'a, N, Ch> =
    LruCacheEntries<'a, PointN<N>, ChunkNode<Ch>, CompressedLocation>;
pub type LruChunkCacheIntoIter<N, Ch> =
    LruCacheIntoIter<PointN<N>, ChunkNode<Ch>, CompressedLocation>;

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
