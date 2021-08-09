use crate::{
    caching::{CacheEntry, LocalCache},
    chunk::{CompressedChunks, LruChunkCacheEntries, LruChunkCacheKeys},
    dev_prelude::{
        Channel, ChunkKey, ChunkMap, ChunkReadStorage, CompressibleChunkStorage, Compression,
        IterChunkKeys,
    },
    SmallKeyBuildHasher,
};

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

impl<'a, N, Compr> ChunkReadStorage<N> for CompressibleChunkStorageReader<'a, N, Compr>
where
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    type Chunk = Compr::Data;

    #[inline]
    fn get(&self, key: ChunkKey<N>) -> Option<&Compr::Data> {
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
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    type Iter = LruChunkCacheKeys<'a, N, Compr::Data>;

    fn chunk_keys(&'a self) -> Self::Iter {
        self.storage.cache.keys()
    }
}

impl<'a, N, Compr> IntoIterator for &'a CompressibleChunkStorageReader<'a, N, Compr>
where
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    type IntoIter = CompressibleChunkStorageReaderIter<'a, N, Compr>;
    type Item = (&'a ChunkKey<N>, &'a Compr::Data);

    fn into_iter(self) -> Self::IntoIter {
        let &CompressibleChunkStorageReader {
            storage,
            local_cache,
        } = self;

        CompressibleChunkStorageReaderIter {
            cache_entries: storage.cache.entries(),
            local_cache,
            compressed: &storage.compressed,
        }
    }
}

pub struct CompressibleChunkStorageReaderIter<'a, N, Compr>
where
    Compr: Compression,
{
    cache_entries: LruChunkCacheEntries<'a, N, Compr::Data>,
    local_cache: &'a LocalChunkCache<N, Compr::Data>,
    compressed: &'a CompressedChunks<Compr>,
}

impl<'a, N, Compr> Iterator for CompressibleChunkStorageReaderIter<'a, N, Compr>
where
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    type Item = (&'a ChunkKey<N>, &'a Compr::Data);

    fn next(&mut self) -> Option<Self::Item> {
        self.cache_entries
            .next()
            .map(move |(key, entry)| match entry {
                CacheEntry::Cached(chunk) => (key, chunk),
                CacheEntry::Evicted(location) => (
                    key,
                    self.local_cache.get_or_insert_with(key.clone(), || {
                        self.compressed.get(location.0).unwrap().decompress()
                    }),
                ),
            })
    }
}

/// A `LocalCache` of chunks.
pub type LocalChunkCache<N, Ch> = LocalCache<ChunkKey<N>, Ch, SmallKeyBuildHasher>;
/// A `LocalCache` of 2D chunks.
pub type LocalChunkCache2<Ch> = LocalChunkCache<[i32; 2], Ch>;
/// A `LocalCache` of 3D chunks.
pub type LocalChunkCache3<Ch> = LocalChunkCache<[i32; 3], Ch>;

/// A `ChunkMap` backed by a `CompressibleChunkStorageReader`.
pub type CompressibleChunkMapReader<'a, N, T, Bldr, Compr> =
    ChunkMap<N, T, Bldr, CompressibleChunkStorageReader<'a, N, Compr>>;

pub mod multichannel_aliases {
    use super::*;
    use crate::dev_prelude::{ChunkMapBuilderNxM, FastArrayCompression, FastChannelsCompression};

    macro_rules! compressible_map_reader_type_alias {
        ($name:ident, $dim:ty, $( $chan:ident ),+ ) => {
            pub type $name<'a, By, $( $chan ),+> = CompressibleChunkMapReader<
                'a,
                $dim,
                ($($chan,)+),
                ChunkMapBuilderNxM<$dim, ($($chan,)+), ($(Channel<$chan>,)+)>,
                FastArrayCompression<$dim, FastChannelsCompression<By, ($(Channel<$chan>,)+)>>,
            >;
        };
    }

    pub type CompressibleChunkMapReaderNx1<'a, N, By, A> = CompressibleChunkMapReader<
        'a,
        N,
        A,
        ChunkMapBuilderNxM<N, A, Channel<A>>,
        FastArrayCompression<N, FastChannelsCompression<By, Channel<A>>>,
    >;

    pub type CompressibleChunkMapReader2x1<'a, By, A> =
        CompressibleChunkMapReaderNx1<'a, [i32; 2], By, A>;
    compressible_map_reader_type_alias!(CompressibleChunkMapReader2x2, [i32; 2], A, B);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader2x3, [i32; 2], A, B, C);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader2x4, [i32; 2], A, B, C, D);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader2x5, [i32; 2], A, B, C, D, E);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader2x6, [i32; 2], A, B, C, D, E, F);

    pub type CompressibleChunkMapReader3x1<'a, By, A> =
        CompressibleChunkMapReaderNx1<'a, [i32; 3], By, A>;
    compressible_map_reader_type_alias!(CompressibleChunkMapReader3x2, [i32; 3], A, B);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader3x3, [i32; 3], A, B, C);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader3x4, [i32; 3], A, B, C, D);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader3x5, [i32; 3], A, B, C, D, E);
    compressible_map_reader_type_alias!(CompressibleChunkMapReader3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;
