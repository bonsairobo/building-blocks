use crate::{
    BytesCompression, CacheEntry, Chunk, ChunkMap, ChunkWriteStorage, Compressed,
    CompressibleChunkMapReader, CompressibleChunkStorageReader, Compression, FastChunkCompression,
    FnvLruCache, IterChunkKeys, LocalChunkCache, LruCacheEntries, LruCacheIntoIter, LruCacheKeys,
    MaybeCompressed,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use slab::Slab;

/// A two-tier chunk storage. The first tier is an LRU cache of uncompressed chunks. The second tier is a `Slab` of compressed
/// chunks.
pub struct CompressibleChunkStorage<N, Compr>
where
    Compr: Compression,
{
    pub cache: FnvLruCache<PointN<N>, Compr::Data, CompressedLocation>,
    pub compression: Compr,
    pub compressed: CompressedChunks<Compr>,
}

pub type CompressedChunks<Compr> = Slab<Compressed<Compr>>;

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    pub fn new(compression: Compr) -> Self {
        Self {
            cache: FnvLruCache::default(),
            compression,
            compressed: Slab::new(),
        }
    }
}

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    /// Returns a reader that implements `ChunkReadStorage`.
    pub fn reader<'a>(
        &'a self,
        local_cache: &'a LocalChunkCache<N, Compr::Data>,
    ) -> CompressibleChunkStorageReader<'a, N, Compr> {
        CompressibleChunkStorageReader {
            storage: self,
            local_cache,
        }
    }

    /// Returns a copy of the `Chunk` at `key`.
    ///
    /// WARNING: the cache will not be updated. This method should be used for a read-modify-write workflow where it would be
    /// inefficient to cache the chunk only for it to be overwritten by the modified version.
    pub fn copy_without_caching(
        &self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<Compr::Data, Compressed<Compr>>>
    where
        Compr::Data: Clone,
        Compressed<Compr>: Clone,
    {
        self.cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressed::Decompressed(chunk.clone()),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(self.compressed.get(location.0).unwrap().clone())
            }
        })
    }

    /// Remove the `Chunk` at `key`.
    pub fn remove(
        &mut self,
        key: PointN<N>,
    ) -> Option<MaybeCompressed<Compr::Data, Compressed<Compr>>> {
        self.cache.remove(&key).map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressed::Decompressed(chunk),
            CacheEntry::Evicted(location) => {
                MaybeCompressed::Compressed(self.compressed.remove(location.0))
            }
        })
    }

    /// Compress the least-recently-used, cached chunk. On access, compressed chunks will be
    /// decompressed and cached.
    pub fn compress_lru(&mut self) {
        let compressed_entry = self.compressed.vacant_entry();
        if let Some((_, lru_chunk)) = self
            .cache
            .evict_lru(CompressedLocation(compressed_entry.key()))
        {
            compressed_entry.insert(self.compression.compress(&lru_chunk));
        }
    }

    /// Remove the least-recently-used, cached chunk.
    ///
    /// This is useful for removing a batch of chunks at a time before compressing them in parallel. Then call
    /// `insert_compressed`.
    pub fn remove_lru(&mut self) -> Option<(PointN<N>, Compr::Data)> {
        self.cache.remove_lru()
    }

    /// Insert a compressed chunk. Returns the old chunk if one exists.
    pub fn insert_compressed(
        &mut self,
        key: PointN<N>,
        compressed_chunk: Compressed<Compr>,
    ) -> Option<MaybeCompressed<Compr::Data, Compressed<Compr>>> {
        let compressed_entry = self.compressed.vacant_entry();
        let old_entry = self
            .cache
            .evict(key, CompressedLocation(compressed_entry.key()));
        compressed_entry.insert(compressed_chunk);

        old_entry.map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressed::Decompressed(chunk),
            CacheEntry::Evicted(old_location) => {
                MaybeCompressed::Compressed(self.compressed.remove(old_location.0))
            }
        })
    }

    /// Consumes and flushes the chunk cache into the chunk map. This is not strictly necessary, but
    /// it will help with caching efficiency.
    pub fn flush_local_cache(&mut self, local_cache: LocalChunkCache<N, Compr::Data>) {
        for (key, chunk) in local_cache.into_iter() {
            self.insert_chunk(key, chunk);
        }
    }

    /// Inserts `chunk` at `key` and returns the old chunk.
    pub fn insert_chunk(
        &mut self,
        key: PointN<N>,
        chunk: Compr::Data,
    ) -> Option<MaybeCompressed<Compr::Data, Compressed<Compr>>> {
        self.cache
            .insert(key, chunk)
            .map(|old_entry| match old_entry {
                CacheEntry::Cached(old_chunk) => MaybeCompressed::Decompressed(old_chunk),
                CacheEntry::Evicted(location) => {
                    MaybeCompressed::Compressed(self.compressed.remove(location.0))
                }
            })
    }
}

impl<N, Compr> ChunkWriteStorage<N, Compr::Data> for CompressibleChunkStorage<N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    #[inline]
    fn get_mut(&mut self, key: PointN<N>) -> Option<&mut Compr::Data> {
        let Self {
            cache, compressed, ..
        } = self;

        cache.get_mut_or_repopulate_with(key, |location| compressed.remove(location.0).decompress())
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Compr::Data,
    ) -> &mut Compr::Data {
        let Self {
            cache, compressed, ..
        } = self;
        cache.get_mut_or_insert_with(
            key,
            |location| compressed.remove(location.0).decompress(),
            create_chunk,
        )
    }

    #[inline]
    fn replace(&mut self, key: PointN<N>, chunk: Compr::Data) -> Option<Compr::Data> {
        self.insert_chunk(key, chunk)
            .map(|old_chunk| match old_chunk {
                MaybeCompressed::Decompressed(old_chunk) => old_chunk,
                MaybeCompressed::Compressed(old_chunk) => old_chunk.decompress(),
            })
    }

    #[inline]
    fn write(&mut self, key: PointN<N>, chunk: Compr::Data) {
        self.insert_chunk(key, chunk);
    }
}

impl<'a, N, Compr> IterChunkKeys<'a, N> for CompressibleChunkStorage<N, Compr>
where
    N: 'a,
    PointN<N>: Hash + IntegerPoint<N>,
    Compr::Data: 'a,
    Compr: Compression,
{
    type Iter = LruChunkCacheKeys<'a, N, Compr::Data>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.cache.keys()
    }
}

impl<N, Compr> IntoIterator for CompressibleChunkStorage<N, Compr>
where
    N: 'static,
    Compr: 'static + Compression,
{
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;
    type Item = (PointN<N>, Compr::Data);

    fn into_iter(self) -> Self::IntoIter {
        let Self {
            cache,
            mut compressed,
            ..
        } = self;

        Box::new(cache.into_iter().map(move |(key, entry)| match entry {
            CacheEntry::Cached(chunk) => (key, chunk),
            CacheEntry::Evicted(location) => (key, compressed.remove(location.0).decompress()),
        }))
    }
}

/// A `ChunkMap` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkMap<N, T, Meta, B> =
    ChunkMap<N, T, Meta, CompressibleChunkStorage<N, FastChunkCompression<N, T, Meta, B>>>;

impl<N, T, Meta, B> CompressibleChunkMap<N, T, Meta, B>
where
    PointN<N>: Hash + IntegerPoint<N>,
    T: 'static + Copy,
    Meta: Clone,
    B: BytesCompression,
{
    /// Construct a reader for this map.
    pub fn reader<'a>(
        &'a self,
        local_cache: &'a LocalChunkCache<N, Chunk<N, T, Meta>>,
    ) -> CompressibleChunkMapReader<N, T, Meta, B> {
        self.builder()
            .build_with_read_storage(self.storage().reader(local_cache))
    }
}

/// An index into a compressed chunk slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedLocation(pub usize);

pub type LruChunkCacheKeys<'a, N, Ch> = LruCacheKeys<'a, PointN<N>, Ch, CompressedLocation>;
pub type LruChunkCacheEntries<'a, N, Ch> = LruCacheEntries<'a, PointN<N>, Ch, CompressedLocation>;
pub type LruChunkCacheIntoIter<N, Ch> = LruCacheIntoIter<PointN<N>, Ch, CompressedLocation>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        /// 2-dimensional `CompressibleChunkStorage`.
        pub type CompressibleChunkStorage2<T, Meta = (), B = $backend> =
            CompressibleChunkStorage<[i32; 2], FastChunkCompression<[i32; 2], T, Meta, B>>;
        /// 3-dimensional `CompressibleChunkStorage`.
        pub type CompressibleChunkStorage3<T, Meta = (), B = $backend> =
            CompressibleChunkStorage<[i32; 3], FastChunkCompression<[i32; 3], T, Meta, B>>;

        /// A 2-dimensional `CompressibleChunkMap`.
        pub type CompressibleChunkMap2<T, Meta = (), B = $backend> =
            CompressibleChunkMap<[i32; 2], T, Meta, B>;
        /// A 3-dimensional `CompressibleChunkMap`.
        pub type CompressibleChunkMap3<T, Meta = (), B = $backend> =
            CompressibleChunkMap<[i32; 3], T, Meta, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);
