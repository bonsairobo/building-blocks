use crate::{
    BytesCompression, CacheEntry, Chunk, ChunkMap, ChunkReadStorage, CompressedChunks,
    CompressibleChunkStorage, IterChunkKeys, LocalCache, LruChunkCacheEntries, LruChunkCacheKeys,
};

use building_blocks_core::prelude::*;
use core::hash::Hash;
use fnv::FnvBuildHasher;

/// An object for reading from `CompressibleChunkStorage` with only `&self`. Easily construct one of these using the
/// `CompressibleChunkStorage::reader` method.
///
/// This works by using a `LocalChunkCache` for storing decompressed `Chunk`s from cache misses.
pub struct CompressibleChunkStorageReader<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N>,
    T: 'static + Copy,
    Meta: Clone,
    B: BytesCompression,
{
    pub storage: &'a CompressibleChunkStorage<N, T, Meta, B>,
    pub local_cache: &'a LocalChunkCache<N, T, Meta>,
}

impl<'a, N, T, Meta, B> ChunkReadStorage<N, T, Meta>
    for CompressibleChunkStorageReader<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N> + Hash + Eq,
    T: Copy,
    Meta: Clone,
    B: BytesCompression,
{
    #[inline]
    fn get(&self, key: PointN<N>) -> Option<&Chunk<N, T, Meta>> {
        let Self {
            storage: CompressibleChunkStorage {
                cache, compressed, ..
            },
            local_cache,
        } = self;

        cache.get(&key).map(|entry| match entry {
            CacheEntry::Cached(value) => value,
            CacheEntry::Evicted(location) => local_cache
                .get_or_insert_with(key, || compressed.get(location.0).unwrap().decompress()),
        })
    }
}

impl<'a, N, T, Meta, B> IterChunkKeys<'a, N> for CompressibleChunkStorageReader<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N> + Hash + Eq,
    T: Copy,
    Meta: Clone,
    B: BytesCompression,
{
    type Iter = LruChunkCacheKeys<'a, N, T, Meta>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.storage.cache.keys()
    }
}

impl<'a, N, T, Meta, B> IntoIterator for &'a CompressibleChunkStorageReader<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N> + Hash + Eq,
    T: 'static + Copy,
    Meta: Clone,
    B: BytesCompression,
{
    type IntoIter = CompressibleChunkStorageReaderIntoIter<'a, N, T, Meta, B>;
    type Item = (&'a PointN<N>, &'a Chunk<N, T, Meta>);

    fn into_iter(self) -> Self::IntoIter {
        let &CompressibleChunkStorageReader {
            storage,
            local_cache,
        } = self;

        CompressibleChunkStorageReaderIntoIter {
            cache_entries: storage.cache.entries(),
            local_cache,
            compressed: &storage.compressed,
        }
    }
}

pub struct CompressibleChunkStorageReaderIntoIter<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N>,
    T: 'static + Copy,
    Meta: Clone,
    B: BytesCompression,
{
    cache_entries: LruChunkCacheEntries<'a, N, T, Meta>,
    local_cache: &'a LocalChunkCache<N, T, Meta>,
    compressed: &'a CompressedChunks<N, T, Meta, B>,
}

impl<'a, N, T, Meta, B> Iterator for CompressibleChunkStorageReaderIntoIter<'a, N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N> + Hash + Eq,
    T: Copy,
    Meta: Clone,
    B: BytesCompression,
{
    type Item = (&'a PointN<N>, &'a Chunk<N, T, Meta>);

    fn next(&mut self) -> Option<Self::Item> {
        self.cache_entries
            .next()
            .map(move |(key, entry)| match entry {
                CacheEntry::Cached(chunk) => (key, chunk),
                CacheEntry::Evicted(location) => (
                    key,
                    self.local_cache.get_or_insert_with(*key, || {
                        self.compressed.get(location.0).unwrap().decompress()
                    }),
                ),
            })
    }
}

/// A `LocalCache` of `Chunk`s.
pub type LocalChunkCache<N, T, Meta = ()> =
    LocalCache<PointN<N>, Chunk<N, T, Meta>, FnvBuildHasher>;
/// A `LocalCache` of `Chunk2`s.
pub type LocalChunkCache2<T, Meta = ()> = LocalChunkCache<[i32; 2], T, Meta>;
/// A `LocalCache` of `Chunk3`s.
pub type LocalChunkCache3<T, Meta = ()> = LocalChunkCache<[i32; 3], T, Meta>;

/// A `ChunkMap` backed by a `CompressibleChunkStorageReader`.
pub type CompressibleChunkMapReader<'a, N, T, Meta, B> =
    ChunkMap<N, T, Meta, CompressibleChunkStorageReader<'a, N, T, Meta, B>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        /// 2-dimensional `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader2<'a, T, Meta = (), B = $backend> =
            CompressibleChunkStorageReader<'a, [i32; 2], T, Meta, B>;
        /// 3-dimensional `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader3<'a, T, Meta = (), B = $backend> =
            CompressibleChunkStorageReader<'a, [i32; 3], T, Meta, B>;

        /// 2-dimensional `CompressibleChunkMapReader`.
        pub type CompressibleChunkMapReader2<'a, T, Meta = (), B = $backend> =
            CompressibleChunkMapReader<'a, [i32; 2], T, Meta, B>;
        /// 3-dimensional `CompressibleChunkMapReader`.
        pub type CompressibleChunkMapReader3<'a, T, Meta = (), B = $backend> =
            CompressibleChunkMapReader<'a, [i32; 3], T, Meta, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);
