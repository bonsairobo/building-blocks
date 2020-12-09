//! A memory-efficient sparse lattice map made of up array chunks.
//!
//! The data can either be addressed by chunk key with the `get_chunk*` methods or by individual
//! points using the `Get*` and `ForEach*` trait impls. The map of chunks uses `Point3i` keys. The
//! key for a chunk is the minimum point in that chunk, which is always a multiple of the chunk
//! shape. Chunk shape dimensions must be powers of 2, which allows for efficiently calculating a
//! chunk key from any point in the chunk.
//!
//! The chunk map supports chunk compression, and an LRU cache stores chunks after they are
//! decompressed. Because reading a chunk may result in decompression, we will eventually want to
//! mutate the cache in order to store that chunk. If you need to read from the map without mutating
//! it, the const `get_chunk*` methods take a `LocalChunkCache`. To read individual points, you can
//! use the `ChunkMapReader`, which also uses a `LocalChunkCache`. A `LocalChunkCache` can be
//! written back to the `ChunkMap` using the `flush_chunk_cache` method.
//!
//! Two compression backends are usable. The default is LZ4, which has the best tradeoff of
//! compression rate and speed. The other is snap, which has a pure-rust implementation. Snap
//! can be used by enabling the "snappy" feature.
//!
//! # Example Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let chunk_shape = PointN([16; 3]); // components must be powers of 2
//! let ambient_value = 0;
//! let default_chunk_meta = (); // chunk metadata is optional
//! let mut map = ChunkMap::new(chunk_shape, ambient_value, default_chunk_meta, Lz4 { level: 10 });
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted and cached.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Maybe we are tight on memory. Repetitive maps are very compressible. In a game
//! // setting, you would probably have a system dedicated to monitoring the memory usage and
//! // compressing chunks when appropriate.
//! map.compress_lru_chunk();
//!
//! // Even though the map is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks.
//! let bounding_extent = map.bounding_extent();
//!
//! // Now we can read back the values without mutating the map by using local caching. Compressed
//! // chunks will be decompressed into our local cache.
//! let local_cache = LocalChunkCache::new();
//! let reader = ChunkMapReader::new(&map, &local_cache);
//! reader.for_each(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, 1);
//!     } else {
//!         // The points that we didn't write explicitly got an ambient value when the chunk was
//!         // inserted. Also any points in `bounding_extent` that don't have a chunk will also take
//!         // the ambient value.
//!         assert_eq!(value, 0);
//!     }
//! });
//!
//! // It's safe to gather up some const chunk references. The reader will reuse our local cache.
//! let mut chunk_refs = Vec::new();
//! for chunk_key in reader.chunk_keys() {
//!     chunk_refs.push(reader.get_chunk(*chunk_key));
//! }
//!
//! // You can also access individual points like you can with a `ArrayN`. This is about
//! // 10x slower than iterating, because it hashes the chunk coordinates for every access.
//! for p in write_points.iter() {
//!     assert_eq!(reader.get(p), 1);
//! }
//! assert_eq!(reader.get(&PointN([1, 1, 1])), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([32; 3]));
//! let mut dense_map = Array3::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &reader, &mut dense_map);
//!
//! // When you're done accessing the map, you should flush you local cache. This is not strictly
//! // necessary, but it makes the caching more efficient.
//! map.flush_chunk_cache(local_cache);
//! ```

mod chunk;
mod reader;

pub use chunk::*;
pub use reader::*;

use crate::{
    access::{ForEachMut, GetUncheckedMutRelease, WriteExtent},
    array::{Array, ArrayN},
    GetMut,
};

use building_blocks_core::{
    bounding_extent, ComponentwiseIntegerOps, ExtentN, IntegerExtent, IntegerPoint, Ones, Point2i,
    Point3i, PointN,
};

use compressible_map::{
    BincodeCompression, BytesCompression, Compressed, CompressibleMap, Compression, LocalCache,
    MaybeCompressed,
};
use core::hash::Hash;
use core::ops::{Div, Mul};
use fnv::FnvHashMap;
use futures::future::join_all;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// We only use these for adding convenient type aliases.
#[cfg(all(feature = "lz4", not(feature = "snappy")))]
use compressible_map::Lz4;
#[cfg(all(not(feature = "lz4"), feature = "snappy"))]
use compressible_map::Snappy;

// LZ4 and Snappy are not mutually exclusive, but if you only use one, then you want to have these
// aliases refer to the choice you made.
#[cfg(all(feature = "lz4", not(feature = "snappy")))]
pub mod conditional_aliases {
    use super::*;

    pub use chunk::conditional_aliases::*;
    pub use reader::conditional_aliases::*;

    pub type ChunkMap2<T, M = (), B = Lz4> = ChunkMap<[i32; 2], T, M, B>;
    pub type ChunkMap3<T, M = (), B = Lz4> = ChunkMap<[i32; 3], T, M, B>;
    pub type SerializableChunkMap2<T, M = (), B = Lz4> = SerializableChunkMap<[i32; 2], T, M, B>;
    pub type SerializableChunkMap3<T, M = (), B = Lz4> = SerializableChunkMap<[i32; 3], T, M, B>;
}
#[cfg(all(not(feature = "lz4"), feature = "snappy"))]
pub mod conditional_aliases {
    use super::*;

    pub use chunk::conditional_aliases::*;
    pub use reader::conditional_aliases::*;

    pub type ChunkMap2<T, M = (), B = Snappy> = ChunkMap<[i32; 2], T, M, B>;
    pub type ChunkMap3<T, M = (), B = Snappy> = ChunkMap<[i32; 3], T, M, B>;
    pub type SerializableChunkMap2<T, M = (), B = Snappy> = SerializableChunkMap<[i32; 2], T, M, B>;
    pub type SerializableChunkMap3<T, M = (), B = Snappy> = SerializableChunkMap<[i32; 3], T, M, B>;
}

/// A lattice map made up of same-shaped `ArrayN` chunks. It takes a value at every possible
/// `PointN`, because accesses made outside of the stored chunks will return some ambient value
/// specified on creation.
///
/// See the [module-level docs](../chunk_map/index.html) for more details and examples.
pub struct ChunkMap<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    ExtentN<N>: IntegerExtent<N>,
{
    chunk_shape: PointN<N>,
    chunk_shape_mask: PointN<N>,
    chunk_shape_log2: PointN<N>,

    // The value to use when none is specified, i.e. when filling new chunks or erasing points.
    ambient_value: T,

    default_chunk_metadata: M,

    /// The chunks themselves, stored in a `CompressibleMap`.
    ///
    /// SAFETY: Don't mutate this directly unless you know what you're doing.
    pub chunks: CompressibleFnvMap<PointN<N>, Chunk<N, T, M>, FastChunkCompression<N, T, M, B>>,
}

type CompressibleFnvMap<K, V, A> = CompressibleMap<K, V, A, fnv::FnvBuildHasher>;

pub type LocalChunkCache<N, T, M> = LocalCache<PointN<N>, Chunk<N, T, M>, fnv::FnvBuildHasher>;
pub type LocalChunkCache2<T, M = ()> =
    LocalCache<Point2i, Chunk<[i32; 2], T, M>, fnv::FnvBuildHasher>;
pub type LocalChunkCache3<T, M = ()> =
    LocalCache<Point3i, Chunk<[i32; 3], T, M>, fnv::FnvBuildHasher>;

impl<N, T, M, B> ChunkMap<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    /// Creates an empty map.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    pub fn new(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: M,
        compression_params: B,
    ) -> Self {
        Self {
            chunk_shape,
            chunk_shape_mask: chunk_shape.mask(),
            chunk_shape_log2: chunk_shape.ilog2(),
            ambient_value,
            default_chunk_metadata,
            chunks: CompressibleFnvMap::new(FastChunkCompression::new(compression_params)),
        }
    }

    /// Determines whether `key` is a valid chunk key. This means it must be a multiple of the chunk
    /// shape.
    pub fn chunk_key_is_valid(&self, key: &PointN<N>) -> bool {
        self.chunk_shape.mul(key.div(self.chunk_shape)).eq(key)
    }

    /// The constant shape of a chunk. The same for all chunks.
    pub fn chunk_shape(&self) -> &PointN<N> {
        &self.chunk_shape
    }

    /// The mask used for calculating the chunk key of a chunk that contains a given point.
    pub fn chunk_shape_mask(&self) -> &PointN<N> {
        &self.chunk_shape_mask
    }

    /// Returns the key of the chunk that contains `point`.
    pub fn chunk_key_containing_point(&self, point: &PointN<N>) -> PointN<N> {
        PointN::chunk_key_containing_point(self.chunk_shape_mask(), point)
    }

    /// Returns an iterator over all chunk keys for chunks that overlap the given extent.
    pub fn chunk_keys_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        chunk_keys_for_extent(self.chunk_shape_log2, extent)
    }

    /// The extent spanned by the chunk at `key`.
    pub fn extent_for_chunk_at_key(&self, key: &PointN<N>) -> ExtentN<N> {
        extent_for_chunk_at_key(&self.chunk_shape, key)
    }

    /// Insert a chunk at `key`. The chunk must have the same shape as `Self::chunk_shape`, and the
    /// key must be a multiple of the chunk shape. These assertions will be made in debug mode.
    pub fn insert_chunk(
        &mut self,
        key: PointN<N>,
        chunk: Chunk<N, T, M>,
    ) -> Option<MaybeCompressedChunk<N, T, M, B>> {
        debug_assert!(chunk.array.extent().shape.eq(self.chunk_shape()));
        debug_assert!(self.chunk_key_is_valid(&key));

        self.chunks.insert(key, chunk)
    }

    /// Remove the chunk at `key`. The key must be a multiple of the chunk shape. This will be
    /// asserted win debug mode.
    pub fn remove_chunk(&mut self, key: PointN<N>) -> Option<MaybeCompressedChunk<N, T, M, B>> {
        debug_assert!(self.chunk_key_is_valid(&key));

        self.chunks.remove(&key)
    }

    /// Returns the chunk at `key` if it exists.
    pub fn get_chunk<'a>(
        &'a self,
        key: PointN<N>,
        local_cache: &'a LocalChunkCache<N, T, M>,
    ) -> Option<&Chunk<N, T, M>> {
        debug_assert!(self.chunk_key_is_valid(&key));

        self.chunks.get_const(key, local_cache)
    }

    /// Returns a copy of the chunk at `key`.
    ///
    /// WARNING: the cache will not be updated. This method should be used for a read-modify-write
    /// workflow where it would be inefficient to cache the chunk only for it to be overwritten by
    /// the modified version.
    pub fn copy_chunk_without_caching(
        &self,
        key: &PointN<N>,
    ) -> Option<MaybeCompressedChunk<N, T, M, B>>
    where
        Chunk<N, T, M>: Clone,
        Compressed<FastChunkCompression<N, T, M, B>>: Clone,
    {
        debug_assert!(self.chunk_key_is_valid(key));

        self.chunks.get_copy_without_caching(key)
    }

    /// Returns the mutable chunk at `key` if it exists.
    pub fn get_mut_chunk(&mut self, key: PointN<N>) -> Option<&mut Chunk<N, T, M>> {
        debug_assert!(self.chunk_key_is_valid(&key));

        self.chunks.get_mut(key)
    }

    /// Get mutable chunk for `key`. If `key` does not exist, calls `fill_empty_chunk` to fill that
    /// entry first.
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl Fn(&PointN<N>, &ExtentN<N>) -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M> {
        debug_assert!(self.chunk_key_is_valid(&key));

        let ChunkMap {
            chunk_shape,
            chunks,
            ..
        } = self;

        chunks.get_or_insert_with(key, || {
            create_chunk(&key, &extent_for_chunk_at_key(chunk_shape, &key))
        })
    }

    /// Returns the chunk containing `point` if it exists.
    pub fn get_chunk_containing_point<'a>(
        &'a self,
        point: &PointN<N>,
        local_cache: &'a LocalChunkCache<N, T, M>,
    ) -> Option<(PointN<N>, &Chunk<N, T, M>)> {
        let chunk_key = self.chunk_key_containing_point(point);

        self.get_chunk(chunk_key, local_cache)
            .map(|c| (chunk_key, c))
    }

    /// Returns the mutable chunk containing `point` if it exists.
    pub fn get_mut_chunk_containing_point(
        &mut self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &mut Chunk<N, T, M>)> {
        let chunk_key = self.chunk_key_containing_point(point);

        self.get_mut_chunk(chunk_key).map(|c| (chunk_key, c))
    }

    /// An iterator over all occupied chunk keys.
    pub fn chunk_keys(&self) -> impl Iterator<Item = &PointN<N>> {
        self.chunks.keys()
    }

    /// The smallest extent that bounds all chunks.
    pub fn bounding_extent(&self) -> ExtentN<N> {
        bounding_extent(self.chunks.keys().flat_map(|key| {
            let chunk_extent = self.extent_for_chunk_at_key(key);

            vec![chunk_extent.minimum, chunk_extent.max()].into_iter()
        }))
    }

    /// Get mutable data for point `p`. If `p` does not exist, calls `fill_empty_chunk` to fill
    /// that entry first.
    pub fn get_mut_or_insert_chunk_with(
        &mut self,
        p: &PointN<N>,
        create_chunk: impl Fn(&PointN<N>, &ExtentN<N>) -> Chunk<N, T, M>,
    ) -> (PointN<N>, &mut T)
    where
        ArrayN<N, T>: Array<N>,
    {
        let key = self.chunk_key_containing_point(p);
        let chunk = self.get_mut_chunk_or_insert_with(key, create_chunk);

        (key, chunk.array.get_unchecked_mut_release(p))
    }

    /// Sets point `p` to value `T`. If `p` is in a chunk that doesn't exist yet, then the chunk
    /// will first be filled with the ambient value and default metadata.
    pub fn get_mut_and_key(&mut self, p: &PointN<N>) -> (PointN<N>, &mut T)
    where
        ArrayN<N, T>: Array<N>,
    {
        let key = self.chunk_key_containing_point(p);
        let ChunkMap {
            chunk_shape,
            ambient_value,
            default_chunk_metadata,
            chunks,
            ..
        } = self;
        let array = &mut chunks
            .get_or_insert_with(key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(extent_for_chunk_at_key(chunk_shape, &key), *ambient_value),
            })
            .array;

        (key, array.get_unchecked_mut_release(p))
    }

    /// Compress the least-recently-used, cached chunk. On access, compressed chunks will be
    /// decompressed and cached.
    pub fn compress_lru_chunk(&mut self) {
        self.chunks.compress_lru();
    }

    /// Remove the least-recently-used, cached chunk.
    pub fn remove_lru_chunk(&mut self) -> Option<(PointN<N>, Chunk<N, T, M>)> {
        self.chunks.remove_lru()
    }

    /// Consumes and flushes the chunk cache into the chunk map. This is not strictly necessary, but
    /// it will help with caching efficiency.
    pub fn flush_chunk_cache(&mut self, local_cache: LocalChunkCache<N, T, M>) {
        self.chunks.flush_local_cache(local_cache);
    }

    /// Return an iterator over all chunk references.
    pub fn chunk_iter(
        &self,
    ) -> impl Iterator<Item = (&PointN<N>, MaybeCompressedChunkRef<N, T, M, B>)> {
        self.chunks.iter()
    }

    /// Consume `self` and return an iterator over all chunks.
    pub fn into_chunk_iter(
        self,
    ) -> impl Iterator<Item = (PointN<N>, MaybeCompressedChunk<N, T, M, B>)> {
        self.chunks.into_iter()
    }

    /// Return the ambient value of the map.
    #[inline]
    pub fn ambient_value(&self) -> T {
        self.ambient_value
    }

    /// Return the default metadata for chunks.
    #[inline]
    pub fn default_chunk_metadata(&self) -> &M {
        &self.default_chunk_metadata
    }

    /// Returns a serializable version of this map. This will compress every chunk in a portable
    /// way.
    pub async fn to_serializable(
        &self,
        params: BincodeCompression<Chunk<N, T, M>, B>,
    ) -> SerializableChunkMap<N, T, M, B>
    where
        Chunk<N, T, M>: DeserializeOwned + Serialize,
        BincodeCompression<Chunk<N, T, M>, B>: Copy, // TODO: this should be inferred
    {
        let chunk_futures: Vec<_> = self
            .chunks
            .iter()
            .map(|(chunk_key, chunk)| async move {
                let portable_chunk = match chunk {
                    MaybeCompressed::Compressed(compressed_chunk) => {
                        params.compress(&compressed_chunk.decompress())
                    }
                    MaybeCompressed::Decompressed(chunk) => params.compress(chunk),
                };

                (*chunk_key, portable_chunk)
            })
            .collect();

        let compressed_chunks = join_all(chunk_futures).await.into_iter().collect();

        SerializableChunkMap {
            chunk_shape: self.chunk_shape,
            ambient_value: self.ambient_value,
            default_chunk_metadata: self.default_chunk_metadata.clone(),
            compressed_chunks,
        }
    }

    /// Returns a new map from the serialized, compressed version. This will decompress each chunk
    /// and compress it again, but in a faster format.
    pub async fn from_serializable(map: &SerializableChunkMap<N, T, M, B>, params: B) -> Self
    where
        Chunk<N, T, M>: DeserializeOwned + Serialize,
        FastChunkCompression<N, T, M, B>: Copy, // TODO: should be inferred
    {
        let params = FastChunkCompression::new(params);
        let all_futures: Vec<_> = map
            .compressed_chunks
            .iter()
            .map(|(chunk_key, compressed_chunk)| async move {
                let decompressed = compressed_chunk.decompress();
                let recompressed = params.compress(&decompressed);

                (*chunk_key, recompressed)
            })
            .collect();

        let all_compressed = join_all(all_futures).await.into_iter().collect();

        Self {
            chunk_shape: map.chunk_shape,
            chunk_shape_mask: map.chunk_shape.mask(),
            chunk_shape_log2: map.chunk_shape.ilog2(),
            ambient_value: map.ambient_value,
            default_chunk_metadata: map.default_chunk_metadata.clone(),
            chunks: CompressibleFnvMap::from_all_compressed(params, all_compressed),
        }
    }
}

impl<N, T, M, B> ChunkMap<N, T, M, B>
where
    Self: ForEachMut<N, PointN<N>, Data = T>,
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T) {
        self.for_each_mut(extent, |_p, v| *v = value);
    }
}

/// Call `ChunkMap::to_serializable` to get this type, which is an compressed, serde-serializable
/// type.
#[derive(Deserialize, Serialize)]
pub struct SerializableChunkMap<N, T, M, B>
where
    B: BytesCompression,
    Chunk<N, T, M>: DeserializeOwned + Serialize,
    PointN<N>: Eq + Hash,
{
    pub chunk_shape: PointN<N>,
    pub ambient_value: T,
    pub default_chunk_metadata: M,
    pub compressed_chunks: FnvHashMap<PointN<N>, BincodeCompressedChunk<N, T, M, B>>,
}

/// Returns the extent of the chunk at `key`.
pub fn extent_for_chunk_at_key<N>(chunk_shape: &PointN<N>, key: &PointN<N>) -> ExtentN<N>
where
    PointN<N>: Copy,
{
    ExtentN::from_min_and_shape(*key, *chunk_shape)
}

/// Returns an iterator over all chunk keys for chunks that overlap the given extent.
pub fn chunk_keys_for_extent<N>(
    chunk_shape_log2: PointN<N>,
    extent: &ExtentN<N>,
) -> impl Iterator<Item = PointN<N>>
where
    PointN<N>: IntegerPoint + Ones,
    ExtentN<N>: IntegerExtent<N>,
{
    let key_min = extent.minimum.vector_right_shift(&chunk_shape_log2);
    let key_max = extent.max().vector_right_shift(&chunk_shape_log2);

    ExtentN::from_min_and_max(key_min, key_max)
        .iter_points()
        .map(move |p| p.vector_left_shift(&chunk_shape_log2))
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, M, B> GetMut<&PointN<N>> for ChunkMap<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N>,
{
    type Data = T;

    fn get_mut(&mut self, p: &PointN<N>) -> &mut T {
        let (_chunk_key, value_mut) = self.get_mut_and_key(p);

        value_mut
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<'a, N, T, M, B> ForEachMut<N, PointN<N>> for ChunkMap<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: ForEachMut<N, PointN<N>, Data = T>,
{
    type Data = T;

    fn for_each_mut(&mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &mut Self::Data)) {
        let ChunkMap {
            chunk_shape,
            chunk_shape_log2,
            ambient_value,
            default_chunk_metadata,
            chunks,
            ..
        } = self;

        for chunk_key in chunk_keys_for_extent(*chunk_shape_log2, extent) {
            let chunk = chunks.get_or_insert_with(chunk_key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(
                    extent_for_chunk_at_key(chunk_shape, &chunk_key),
                    *ambient_value,
                ),
            });
            chunk.array.for_each_mut(extent, |p, value| f(p, value));
        }
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

// If ArrayN supports writing from type Src, then so does ChunkMap.
impl<'a, N, T, M, B, Src> WriteExtent<N, Src> for ChunkMap<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    Src: Copy,
    PointN<N>: IntegerPoint + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: WriteExtent<N, Src>,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        let ChunkMap {
            chunk_shape,
            chunk_shape_log2,
            ambient_value,
            default_chunk_metadata,
            chunks,
            ..
        } = self;

        for chunk_key in chunk_keys_for_extent(*chunk_shape_log2, extent) {
            let chunk = chunks.get_or_insert_with(chunk_key, || Chunk {
                metadata: default_chunk_metadata.clone(),
                array: ArrayN::fill(
                    extent_for_chunk_at_key(chunk_shape, &chunk_key),
                    *ambient_value,
                ),
            });
            chunk.array.write_extent(extent, src);
        }
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(all(test, feature = "lz4"))]
mod tests {
    use super::*;

    use crate::{access::Get, copy_extent, Array3, Lz4};

    use building_blocks_core::Extent3i;

    #[test]
    fn chunk_keys_for_extent_gives_keys_for_chunks_overlapping_extent() {
        let chunk_shape = PointN([16; 3]);
        let query_extent = Extent3i::from_min_and_shape(PointN([15; 3]), PointN([16; 3]));
        let chunk_keys: Vec<_> =
            chunk_keys_for_extent(chunk_shape.ilog2(), &query_extent).collect();

        assert_eq!(
            chunk_keys,
            vec![
                PointN([0, 0, 0]),
                PointN([16, 0, 0]),
                PointN([0, 16, 0]),
                PointN([16, 16, 0]),
                PointN([0, 0, 16]),
                PointN([16, 0, 16]),
                PointN([0, 16, 16]),
                PointN([16, 16, 16])
            ]
        );
    }

    #[test]
    fn write_and_read_points() {
        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::new(chunk_shape, ambient_value, (), Lz4 { level: 10 });

        let points = [
            [0, 0, 0],
            [1, 2, 3],
            [16, 0, 0],
            [0, 16, 0],
            [0, 0, 16],
            [15, 0, 0],
            [-15, 0, 0],
        ];

        for p in points.iter().cloned() {
            assert_eq!(map.get_mut(&PointN(p)), &mut 0);
            *map.get_mut(&PointN(p)) = 1;
            assert_eq!(map.get_mut(&PointN(p)), &mut 1);
        }
    }

    #[test]
    fn write_extent_with_for_each_then_read() {
        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::new(chunk_shape, ambient_value, (), Lz4 { level: 10 });

        let write_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        map.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        let local_cache = LocalChunkCache3::new();
        let reader = ChunkMapReader::new(&map, &local_cache);
        for p in read_extent.iter_points() {
            if write_extent.contains(&p) {
                assert_eq!(reader.get(&p), 1);
            } else {
                assert_eq!(reader.get(&p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        let array = Array3::fill(extent_to_copy, 1);

        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap::new(chunk_shape, ambient_value, (), Lz4 { level: 10 });

        copy_extent(&extent_to_copy, &array, &mut map);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        let local_cache = LocalChunkCache3::new();
        let reader = ChunkMapReader::new(&map, &local_cache);
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(&p) {
                assert_eq!(reader.get(&p), 1);
            } else {
                assert_eq!(reader.get(&p), 0);
            }
        }
    }
}
