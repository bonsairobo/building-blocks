use crate::{
    ArrayN, CacheEntry, ChunkMap, ChunkReadStorage, CompressedChunks, CompressibleChunkStorage,
    Compression, FastArrayCompression, IterChunkKeys, LocalCache, LruChunkCacheEntries,
    LruChunkCacheKeys,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvBuildHasher;

/// An object for reading from `CompressibleChunkStorage` with only `&self`. Easily construct one of these using the
/// `CompressibleChunkStorage::reader` method.
///
/// This works by using a `LocalChunkCache` for storing decompressed `Chunk`s from cache misses.
pub struct CompressibleChunkStorageReader<'a, N, Compr>
where
    Compr: Compression,
{
    pub storage: &'a CompressibleChunkStorage<N, Compr>,
    pub local_cache: &'a LocalChunkCache<N, Compr::Data>,
}

impl<'a, N, Compr> ChunkReadStorage<N, Compr::Data> for CompressibleChunkStorageReader<'a, N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    #[inline]
    fn get(&self, key: PointN<N>) -> Option<&Compr::Data> {
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

impl<'a, N, Compr> IterChunkKeys<'a, N> for CompressibleChunkStorageReader<'a, N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    type Iter = LruChunkCacheKeys<'a, N, Compr::Data>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.storage.cache.keys()
    }
}

impl<'a, N, Compr> IntoIterator for &'a CompressibleChunkStorageReader<'a, N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    type IntoIter = CompressibleChunkStorageReaderIntoIter<'a, N, Compr>;
    type Item = (&'a PointN<N>, &'a Compr::Data);

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

pub struct CompressibleChunkStorageReaderIntoIter<'a, N, Compr>
where
    Compr: Compression,
{
    cache_entries: LruChunkCacheEntries<'a, N, Compr::Data>,
    local_cache: &'a LocalChunkCache<N, Compr::Data>,
    compressed: &'a CompressedChunks<Compr>,
}

impl<'a, N, Compr> Iterator for CompressibleChunkStorageReaderIntoIter<'a, N, Compr>
where
    PointN<N>: Hash + IntegerPoint<N>,
    Compr: Compression,
{
    type Item = (&'a PointN<N>, &'a Compr::Data);

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

/// A `LocalCache` of chunks.
pub type LocalChunkCache<N, Ch> = LocalCache<PointN<N>, Ch, FnvBuildHasher>;
/// A `LocalCache` of 2D chunks.
pub type LocalChunkCache2<Ch> = LocalChunkCache<[i32; 2], Ch>;
/// A `LocalCache` of 3D chunks.
pub type LocalChunkCache3<Ch> = LocalChunkCache<[i32; 3], Ch>;

/// A `ChunkMap` backed by a `CompressibleChunkStorageReader`.
pub type CompressibleChunkMapReader<'a, N, T, Ch, Compr> =
    ChunkMap<N, T, Ch, CompressibleChunkStorageReader<'a, N, Compr>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        /// N-dimensional, single-channel `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReaderNx1<'a, N, T, B = $backend> =
            CompressibleChunkStorageReader<'a, N, FastArrayCompression<N, T, B>>;
        /// 2-dimensional, single-channel `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader2x1<'a, T, B = $backend> =
            CompressibleChunkStorageReaderNx1<'a, [i32; 2], T, B>;
        /// 3-dimensional, single-channel `CompressibleChunkStorageReader`.
        pub type CompressibleChunkStorageReader3x1<'a, T, B = $backend> =
            CompressibleChunkStorageReaderNx1<'a, [i32; 3], T, B>;

        /// N-dimensional, single-channel `CompressibleChunkMapReader`.
        pub type CompressibleChunkMapReaderNx1<'a, N, T, B = $backend> = ChunkMap<
            N,
            T,
            ArrayN<N, T>,
            CompressibleChunkStorageReader<'a, N, FastArrayCompression<N, T, B>>,
        >;
        /// 2-dimensional, single-channel `CompressibleChunkMapReader`.
        pub type CompressibleChunkMapReader2x1<'a, T, B = $backend> =
            CompressibleChunkMapReaderNx1<'a, [i32; 2], T, B>;
        /// 3-dimensional, single-channel `CompressibleChunkMapReader`.
        pub type CompressibleChunkMapReader3x1<'a, T, B = $backend> =
            CompressibleChunkMapReaderNx1<'a, [i32; 3], T, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these aliases refer to the choice
// you made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);
