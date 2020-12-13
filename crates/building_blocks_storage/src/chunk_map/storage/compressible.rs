use crate::{
    BytesCompression, CacheEntry, Chunk, ChunkMap, ChunkWriteStorage, Compressed,
    CompressibleChunkStorageReader, Compression, FastChunkCompression, FnvLruCache, IterChunkKeys,
    LocalChunkCache, LruCacheEntries, LruCacheIntoIter, LruCacheKeys, MaybeCompressedChunk,
};

use building_blocks_core::prelude::*;
use core::hash::Hash;
use slab::Slab;

/// A two-tier chunk storage. The first tier is an LRU cache of uncompressed `Chunk`s. The second tier is a `Slab` of compressed
/// `Chunk`s.
pub struct CompressibleChunkStorage<N, T, M, B>
where
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    pub cache: FnvLruCache<PointN<N>, Chunk<N, T, M>, CompressedLocation>,
    pub compression: FastChunkCompression<N, T, M, B>,
    pub compressed: CompressedChunks<N, T, M, B>,
}

pub type CompressedChunks<N, T, M, B> = Slab<Compressed<FastChunkCompression<N, T, M, B>>>;

impl<N, T, M, B> CompressibleChunkStorage<N, T, M, B>
where
    PointN<N>: Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    pub fn new(compression: B) -> Self {
        Self {
            cache: FnvLruCache::default(),
            compression: FastChunkCompression::new(compression),
            compressed: Slab::new(),
        }
    }
}

impl<N, T, M, B> CompressibleChunkStorage<N, T, M, B>
where
    PointN<N>: Clone + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    /// Returns a reader that implements `ChunkReadStorage`.
    pub fn reader<'a>(
        &'a self,
        local_cache: &'a LocalChunkCache<N, T, M>,
    ) -> CompressibleChunkStorageReader<'a, N, T, M, B> {
        CompressibleChunkStorageReader {
            storage: self,
            local_cache,
        }
    }

    /// Returns a copy of the `Chunk` at `key`.
    ///
    /// WARNING: the cache will not be updated. This method should be used for a read-modify-write workflow where it would be
    /// inefficient to cache the chunk only for it to be overwritten by the modified version.
    pub fn copy_without_caching(&self, key: &PointN<N>) -> Option<MaybeCompressedChunk<N, T, M, B>>
    where
        Chunk<N, T, M>: Clone,
        Compressed<FastChunkCompression<N, T, M, B>>: Clone,
    {
        self.cache.get(key).map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressedChunk::Decompressed(chunk.clone()),
            CacheEntry::Evicted(location) => {
                MaybeCompressedChunk::Compressed(self.compressed.get(location.0).unwrap().clone())
            }
        })
    }

    /// Remove the `Chunk` at `key`.
    pub fn remove(&mut self, key: &PointN<N>) -> Option<MaybeCompressedChunk<N, T, M, B>> {
        self.cache.remove(key).map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressedChunk::Decompressed(chunk),
            CacheEntry::Evicted(location) => {
                MaybeCompressedChunk::Compressed(self.compressed.remove(location.0))
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
    pub fn remove_lru(&mut self) -> Option<(PointN<N>, Chunk<N, T, M>)> {
        self.cache.remove_lru()
    }

    /// Insert a compressed chunk. Returns the old chunk if one exists.
    pub fn insert_compressed(
        &mut self,
        key: PointN<N>,
        compressed_chunk: Compressed<FastChunkCompression<N, T, M, B>>,
    ) -> Option<MaybeCompressedChunk<N, T, M, B>> {
        let compressed_entry = self.compressed.vacant_entry();
        let old_entry = self
            .cache
            .evict(key, CompressedLocation(compressed_entry.key()));
        compressed_entry.insert(compressed_chunk);

        old_entry.map(|entry| match entry {
            CacheEntry::Cached(chunk) => MaybeCompressedChunk::Decompressed(chunk),
            CacheEntry::Evicted(old_location) => {
                MaybeCompressedChunk::Compressed(self.compressed.remove(old_location.0))
            }
        })
    }

    /// Consumes and flushes the chunk cache into the chunk map. This is not strictly necessary, but
    /// it will help with caching efficiency.
    pub fn flush_local_cache(&mut self, local_cache: LocalChunkCache<N, T, M>) {
        for (key, chunk) in local_cache.into_iter() {
            self.write_chunk(key, chunk);
        }
    }

    pub fn write_chunk(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) {
        if let Some(old_entry) = self.cache.insert(key, chunk) {
            if let CacheEntry::Evicted(location) = old_entry {
                self.compressed.remove(location.0);
            }
        }
    }
}

impl<N, T, M, B> ChunkWriteStorage<N, T, M> for CompressibleChunkStorage<N, T, M, B>
where
    PointN<N>: Copy + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    #[inline]
    fn get_mut(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, M>> {
        let Self {
            cache, compressed, ..
        } = self;
        cache
            .get_mut_or_repopulate_with(*key, |location| compressed.remove(location.0).decompress())
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M> {
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
    fn replace(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) -> Option<Chunk<N, T, M>> {
        self.cache
            .insert(key, chunk)
            .map(|old_entry| match old_entry {
                CacheEntry::Cached(old_chunk) => old_chunk,
                CacheEntry::Evicted(location) => self.compressed.remove(location.0).decompress(),
            })
    }

    #[inline]
    fn write(&mut self, key: PointN<N>, chunk: Chunk<N, T, M>) {
        self.write_chunk(key, chunk);
    }
}

impl<'a, N, T, M, B> IterChunkKeys<'a, N> for CompressibleChunkStorage<N, T, M, B>
where
    PointN<N>: Clone + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    Chunk<N, T, M>: 'a,
    B: BytesCompression,
{
    type Iter = LruChunkCacheKeys<'a, N, T, M>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.cache.keys()
    }
}

impl<N, T, M, B> IntoIterator for CompressibleChunkStorage<N, T, M, B>
where
    PointN<N>: 'static + Clone + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: 'static + Copy,
    M: 'static + Clone,
    B: 'static + BytesCompression,
{
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;
    type Item = (PointN<N>, Chunk<N, T, M>);

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
pub type CompressibleChunkMap<N, T, M, B> = ChunkMap<N, T, M, CompressibleChunkStorage<N, T, M, B>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedLocation(pub usize);

pub type LruChunkCacheKeys<'a, N, T, M> =
    LruCacheKeys<'a, PointN<N>, Chunk<N, T, M>, CompressedLocation>;
pub type LruChunkCacheEntries<'a, N, T, M> =
    LruCacheEntries<'a, PointN<N>, Chunk<N, T, M>, CompressedLocation>;
pub type LruChunkCacheIntoIter<N, T, M> =
    LruCacheIntoIter<PointN<N>, Chunk<N, T, M>, CompressedLocation>;

macro_rules! define_conditional_aliases {
    ($backend:ty) => {
        /// 2-dimensional `CompressibleChunkStorage`.
        pub type CompressibleChunkStorage2<T, M = (), B = $backend> =
            CompressibleChunkStorage<[i32; 2], T, M, B>;
        /// 3-dimensional `CompressibleChunkStorage`.
        pub type CompressibleChunkStorage3<T, M = (), B = $backend> =
            CompressibleChunkStorage<[i32; 3], T, M, B>;

        /// A 2-dimensional `CompressibleChunkMap`.
        pub type CompressibleChunkMap2<T, M = (), B = $backend> =
            CompressibleChunkMap<[i32; 2], T, M, B>;
        /// A 3-dimensional `CompressibleChunkMap`.
        pub type CompressibleChunkMap3<T, M = (), B = $backend> =
            CompressibleChunkMap<[i32; 3], T, M, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
pub mod conditional_aliases {
    use super::*;
    use crate::Lz4;
    define_conditional_aliases!(Lz4);
}
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
pub mod conditional_aliases {
    use super::*;
    use crate::Snappy;
    define_conditional_aliases!(Snappy);
}
