use crate::{
    BytesCompression, CacheEntry, Chunk, ChunkReadStorage, CompressedChunks,
    CompressibleChunkStorage, IterChunkKeys, LocalCache, LruChunkCacheEntries, LruChunkCacheKeys,
};

use building_blocks_core::prelude::*;
use core::hash::Hash;
use fnv::FnvBuildHasher;

/// An object for reading from `CompressibleChunkStorage` with only `&self`. Easily construct one of these using
/// the `CompressibleChunkStorage::reader` method.
///
/// This works by using a `LocalChunkCache` for storing decompressed `Chunk`s from cache misses.
pub struct CompressibleChunkStorageReader<'a, N, T, M, B>
where
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    pub storage: &'a CompressibleChunkStorage<N, T, M, B>,
    pub local_cache: &'a LocalChunkCache<N, T, M>,
}

impl<'a, N, T, M, B> ChunkReadStorage<N, T, M> for CompressibleChunkStorageReader<'a, N, T, M, B>
where
    PointN<N>: IntegerPoint + Copy + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    #[inline]
    fn get(&self, key: &PointN<N>) -> Option<&Chunk<N, T, M>> {
        let Self {
            storage: CompressibleChunkStorage {
                cache, compressed, ..
            },
            local_cache,
        } = self;

        cache.get(key).map(|entry| match entry {
            CacheEntry::Cached(value) => value,
            CacheEntry::Evicted(location) => local_cache
                .get_or_insert_with(*key, || compressed.get(location.0).unwrap().decompress()),
        })
    }
}

impl<'a, N, T, M, B> IterChunkKeys<'a, N> for CompressibleChunkStorageReader<'a, N, T, M, B>
where
    PointN<N>: Clone + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    type Iter = LruChunkCacheKeys<'a, N, T, M>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.storage.cache.keys()
    }
}

impl<'a, N, T, M, B> IntoIterator for &'a CompressibleChunkStorageReader<'a, N, T, M, B>
where
    PointN<N>: Copy + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    type IntoIter = CompressibleChunkStorageReaderIntoIter<'a, N, T, M, B>;
    type Item = (&'a PointN<N>, &'a Chunk<N, T, M>);

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

pub struct CompressibleChunkStorageReaderIntoIter<'a, N, T, M, B>
where
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    cache_entries: LruChunkCacheEntries<'a, N, T, M>,
    local_cache: &'a LocalChunkCache<N, T, M>,
    compressed: &'a CompressedChunks<N, T, M, B>,
}

impl<'a, N, T, M, B> Iterator for CompressibleChunkStorageReaderIntoIter<'a, N, T, M, B>
where
    PointN<N>: Copy + Hash + Eq,
    ExtentN<N>: IntegerExtent<N>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    type Item = (&'a PointN<N>, &'a Chunk<N, T, M>);

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
pub type LocalChunkCache<N, T, M = ()> = LocalCache<PointN<N>, Chunk<N, T, M>, FnvBuildHasher>;
/// A `LocalCache` of `Chunk2`s.
pub type LocalChunkCache2<T, M = ()> = LocalChunkCache<[i32; 2], T, M>;
/// A `LocalCache` of `Chunk3`s.
pub type LocalChunkCache3<T, M = ()> = LocalChunkCache<[i32; 3], T, M>;

macro_rules! define_conditional_aliases {
    ($backend:ty) => {
        /// 2-dimensional `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader2<'a, T, M = (), B = $backend> =
            CompressibleChunkStorageReader<'a, [i32; 2], T, M, B>;
        /// 3-dimensional `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader3<'a, T, M = (), B = $backend> =
            CompressibleChunkStorageReader<'a, [i32; 3], T, M, B>;
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
