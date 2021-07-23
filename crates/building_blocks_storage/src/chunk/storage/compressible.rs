use crate::{
    caching::*,
    compression::MaybeCompressed,
    dev_prelude::{
        ChunkKey, ChunkMap, ChunkMapBuilder, ChunkWriteStorage, Compressed,
        CompressibleChunkMapReader, CompressibleChunkStorageReader, Compression,
        FastArrayCompression, FastChannelsCompression, FromBytesCompression, IterChunkKeys,
        LocalChunkCache,
    },
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
    pub(crate) cache: SmallKeyLruCache<ChunkKey<N>, Compr::Data, CompressedLocation>,
    pub(crate) compression: Compr,
    pub(crate) compressed: CompressedChunks<Compr>,
}

pub type FastCompressibleChunkStorage<N, By, Chan> =
    CompressibleChunkStorage<N, FastArrayCompression<N, FastChannelsCompression<By, Chan>>>;

impl<N, By, Chan> FastCompressibleChunkStorage<N, By, Chan>
where
    PointN<N>: IntegerPoint<N>,
    ChunkKey<N>: Clone + Eq + Hash,
    FastChannelsCompression<By, Chan>: Compression,
{
    pub fn with_bytes_compression(bytes_compression: By) -> Self {
        Self::new(FastArrayCompression::from_bytes_compression(
            bytes_compression,
        ))
    }
}

pub type CompressedChunks<Compr> = Slab<Compressed<Compr>>;

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    Compr: Compression,
{
    pub fn compression(&self) -> &Compr {
        &self.compression
    }
}

impl<N, Compr> CompressibleChunkStorage<N, Compr>
where
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    pub fn new(compression: Compr) -> Self {
        Self {
            cache: SmallKeyLruCache::default(),
            compression,
            compressed: Slab::new(),
        }
    }

    pub fn len_cached(&self) -> usize {
        self.cache.len_cached()
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
        key: ChunkKey<N>,
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
        key: ChunkKey<N>,
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
    pub fn remove_lru(&mut self) -> Option<(ChunkKey<N>, Compr::Data)> {
        self.cache.remove_lru()
    }

    /// Insert a compressed chunk. Returns the old chunk if one exists.
    pub fn insert_compressed(
        &mut self,
        key: ChunkKey<N>,
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
        for (key, chunk) in local_cache.flush_iter() {
            self.insert_chunk(key, chunk);
        }
    }

    /// Inserts `chunk` at `key` and returns the old chunk.
    pub fn insert_chunk(
        &mut self,
        key: ChunkKey<N>,
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
    ChunkKey<N>: Clone + Eq + Hash,
    Compr: Compression,
{
    #[inline]
    fn get_mut(&mut self, key: ChunkKey<N>) -> Option<&mut Compr::Data> {
        let Self {
            cache, compressed, ..
        } = self;

        cache.get_mut_or_repopulate_with(key, |location| compressed.remove(location.0).decompress())
    }

    #[inline]
    fn get_mut_or_insert_with(
        &mut self,
        key: ChunkKey<N>,
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
    fn replace(&mut self, key: ChunkKey<N>, chunk: Compr::Data) -> Option<Compr::Data> {
        self.insert_chunk(key, chunk)
            .map(|old_chunk| match old_chunk {
                MaybeCompressed::Decompressed(old_chunk) => old_chunk,
                MaybeCompressed::Compressed(old_chunk) => old_chunk.decompress(),
            })
    }

    #[inline]
    fn write(&mut self, key: ChunkKey<N>, chunk: Compr::Data) {
        self.insert_chunk(key, chunk);
    }

    #[inline]
    fn delete(&mut self, key: ChunkKey<N>) {
        self.remove(key);
    }

    #[inline]
    fn pop(&mut self, key: ChunkKey<N>) -> Option<Compr::Data> {
        self.remove(key).map(|ch| ch.into_decompressed())
    }
}

impl<'a, N, Compr> IterChunkKeys<'a, N> for CompressibleChunkStorage<N, Compr>
where
    N: 'a,
    ChunkKey<N>: Clone + Eq + Hash,
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
    type Item = (ChunkKey<N>, Compr::Data);

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

impl<N, T, Bldr, Compr> CompressibleChunkMap<N, T, Bldr, Compr>
where
    ChunkKey<N>: Clone + Eq + Hash,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T> + Clone,
    Compr: Compression<Data = Bldr::Chunk>,
{
    /// Construct a reader for this map.
    pub fn reader<'a>(
        &'a self,
        local_cache: &'a LocalChunkCache<N, Bldr::Chunk>,
    ) -> CompressibleChunkMapReader<'a, N, T, Bldr, Compr> {
        self.builder()
            .clone()
            .build_with_read_storage(self.storage().reader(local_cache))
    }
}

/// An index into a compressed chunk slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedLocation(pub usize);

pub type LruChunkCacheKeys<'a, N, Ch> = LruCacheKeys<'a, ChunkKey<N>, Ch, CompressedLocation>;
pub type LruChunkCacheEntries<'a, N, Ch> = LruCacheEntries<'a, ChunkKey<N>, Ch, CompressedLocation>;
pub type LruChunkCacheIntoIter<N, Ch> = LruCacheIntoIter<ChunkKey<N>, Ch, CompressedLocation>;

/// A `ChunkMap` using `CompressibleChunkStorage` as chunk storage.
pub type CompressibleChunkMap<N, T, Bldr, Compr> =
    ChunkMap<N, T, Bldr, CompressibleChunkStorage<N, Compr>>;

pub mod multichannel_aliases {
    use super::*;
    use crate::array::compression::multichannel_aliases::*;
    use crate::dev_prelude::{Channel, ChunkMapBuilderNxM};

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
            pub type $name<By, $( $chan ),+> = CompressibleChunkMap<
                $dim,
                ($($chan,)+),
                ChunkMapBuilderNxM<$dim, ($($chan,)+), ($(Channel<$chan>,)+)>,
                FastArrayCompression<$dim, FastChannelsCompression<By, ($(Channel<$chan>,)+)>>,
            >;
        };
    }

    pub type CompressibleChunkMapNx1<N, By, A> = CompressibleChunkMap<
        N,
        A,
        ChunkMapBuilderNxM<N, A, Channel<A>>,
        FastArrayCompression<N, FastChannelsCompression<By, Channel<A>>>,
    >;

    pub type CompressibleChunkMap2x1<By, A> = CompressibleChunkMapNx1<[i32; 2], By, A>;
    compressible_map_type_alias!(CompressibleChunkMap2x2, [i32; 2], A, B);
    compressible_map_type_alias!(CompressibleChunkMap2x3, [i32; 2], A, B, C);
    compressible_map_type_alias!(CompressibleChunkMap2x4, [i32; 2], A, B, C, D);
    compressible_map_type_alias!(CompressibleChunkMap2x5, [i32; 2], A, B, C, D, E);
    compressible_map_type_alias!(CompressibleChunkMap2x6, [i32; 2], A, B, C, D, E, F);

    pub type CompressibleChunkMap3x1<By, A> = CompressibleChunkMapNx1<[i32; 3], By, A>;
    compressible_map_type_alias!(CompressibleChunkMap3x2, [i32; 3], A, B);
    compressible_map_type_alias!(CompressibleChunkMap3x3, [i32; 3], A, B, C);
    compressible_map_type_alias!(CompressibleChunkMap3x4, [i32; 3], A, B, C, D);
    compressible_map_type_alias!(CompressibleChunkMap3x5, [i32; 3], A, B, C, D, E);
    compressible_map_type_alias!(CompressibleChunkMap3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;
