use crate::{
    ArrayNx1, CacheEntry, ChunkMap, ChunkReadStorage, CompressedChunks, CompressibleChunkStorage,
    Compression, FastArrayCompressionNx1, IterChunkKeys, LocalCache, LruChunkCacheEntries,
    LruChunkCacheKeys, SmallKeyBuildHasher,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;

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
pub type LocalChunkCache<N, Ch> = LocalCache<PointN<N>, Ch, SmallKeyBuildHasher>;
/// A `LocalCache` of 2D chunks.
pub type LocalChunkCache2<Ch> = LocalChunkCache<[i32; 2], Ch>;
/// A `LocalCache` of 3D chunks.
pub type LocalChunkCache3<Ch> = LocalChunkCache<[i32; 3], Ch>;

/// A `ChunkMap` backed by a `CompressibleChunkStorageReader`.
pub type CompressibleChunkMapReader<'a, N, T, Bldr, Compr> =
    ChunkMap<N, T, Bldr, CompressibleChunkStorageReader<'a, N, Compr>>;

/// N-dimensional, single-channel `CompressibleChunkMapReader`.
pub type CompressibleChunkMapReaderNx1<'a, N, By, T> = ChunkMap<
    N,
    T,
    ArrayNx1<N, T>,
    CompressibleChunkStorageReader<'a, N, FastArrayCompressionNx1<N, By, T>>,
>;
/// 2-dimensional, single-channel `CompressibleChunkMapReader`.
pub type CompressibleChunkMapReader2x1<'a, By, T> =
    CompressibleChunkMapReaderNx1<'a, [i32; 2], By, T>;
/// 3-dimensional, single-channel `CompressibleChunkMapReader`.
pub type CompressibleChunkMapReader3x1<'a, By, T> =
    CompressibleChunkMapReaderNx1<'a, [i32; 3], By, T>;
