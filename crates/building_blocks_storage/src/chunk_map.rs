//! A memory-efficient sparse lattice map.
//!
//! # Example Usage
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let chunk_shape = PointN([16; 3]); // components must be powers of 2
//! let ambient_value = 0;
//! let default_chunk_meta = (); // chunk metadata is optional
//! let mut map = ChunkMap3::new(
//!     chunk_shape, ambient_value, default_chunk_meta, FastLz4 { level: 10 }
//! );
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted and cached.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Maybe we are tight on memory. Sparse or repetitive maps are very compressible. In a game
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
//! let local_cache = LocalChunkCache3::new();
//! let reader = ChunkMapReader3::new(&map, &local_cache);
//! reader.for_each_ref(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, &1);
//!     } else {
//!         // The points that we didn't write explicitly got an ambient value when the chunk was
//!         // inserted. Also any points in `bounding_extent` that don't have a chunk will also take
//!         // the ambient value.
//!         assert_eq!(value, &0);
//!     }
//! });
//!
//! // It's perfectly safe to gather up some const chunk references. We can reuse our local cache.
//! let mut chunk_refs = Vec::new();
//! for chunk_key in map.chunk_keys() {
//!     chunk_refs.push(map.get_chunk(*chunk_key, &local_cache));
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
//! // lot of random access, and the `ChunkMap` can't always support that. Instead, assuming that
//! // you can't just use an exact chunk, you can copy an arbitrary extent into a dense map first.
//! // (The copy itself is fast).
//! let query_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([32; 3]));
//! let reader = ChunkMapReader3::new(&map, &local_cache);
//! let mut dense_map = Array3::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &reader, &mut dense_map);
//!
//! // When you're done accessing the map, you should flush you local cache. This is not strictly
//! // necessary, but it makes the caching more efficient.
//! map.flush_chunk_cache(local_cache);
//! ```

use crate::{
    access::{
        ForEachMut, ForEachRef, GetUncheckedMutRelease, GetUncheckedRefRelease, ReadExtent,
        WriteExtent,
    },
    array::{Array, ArrayCopySrc, ArrayN, FastLz4CompressedArrayN},
    FastLz4, Get, GetMut, GetRef,
};

use building_blocks_core::{
    bounding_extent, ExtentN, IntegerExtent, IntegerPoint, Ones, Point, Point2i, Point3i, PointN,
};

use compressible_map::{
    BincodeLz4, BincodeLz4Compressed, Compressible, CompressibleMap, Decompressible, LocalCache,
    MaybeCompressed,
};
use core::hash::Hash;
use either::Either;
use fnv::FnvHashMap;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Stores a partial (sparse) function on the N-dimensional integers (where N=2 or N=3) in
/// same-shaped chunks using a `CompressibleMap`. The data can either be addressed by chunk with the
/// `get_chunk*` methods or by individual points using the `Get*` and `ForEach*` trait impls.
///
/// Because chunks are either cached or compressed, accesses should eventually mutate the cache. If
/// you need to read from the map without mutating it, the const `get_chunk*` methods take a
/// `LocalChunkCache`. To read individual points, you can use the `ChunkMapReader`, which
/// also uses a `LocalChunkCache`. A `LocalChunkCache` can be written back to the
/// `ChunkMap` using the `flush_chunk_cache` method.
pub struct ChunkMap<N, T, M = ()>
where
    T: Copy,
    M: Clone,
    ExtentN<N>: IntegerExtent<N>,
{
    chunk_shape: PointN<N>,
    chunk_shape_mask: PointN<N>,

    // The value to use when none is specified, i.e. when filling new chunks or erasing points.
    ambient_value: T,

    default_chunk_metadata: M,

    /// The chunks themselves, stored in a `CompressibleMap`.
    pub chunks: CompressibleFnvMap<PointN<N>, Chunk<N, T, M>, FastLz4>,
}

pub type ChunkMap2<T, M = ()> = ChunkMap<[i32; 2], T, M>;
pub type ChunkMap3<T, M = ()> = ChunkMap<[i32; 3], T, M>;

type CompressibleFnvMap<K, V, A> = CompressibleMap<K, V, A, fnv::FnvBuildHasher>;

pub type LocalChunkCache<N, T, M = ()> = LocalCache<PointN<N>, Chunk<N, T, M>, fnv::FnvBuildHasher>;
pub type LocalChunkCache2<T, M = ()> =
    LocalCache<Point2i, Chunk<[i32; 2], T, M>, fnv::FnvBuildHasher>;
pub type LocalChunkCache3<T, M = ()> =
    LocalCache<Point3i, Chunk<[i32; 3], T, M>, fnv::FnvBuildHasher>;

/// One piece of the `ChunkMap`. Contains both some generic metadata and the data for each
/// point in the chunk extent.
#[derive(Clone, Deserialize, Serialize)]
pub struct Chunk<N, T, M = ()> {
    pub metadata: M,
    pub array: ArrayN<N, T>,
}

pub type Chunk2<T, M> = Chunk<[i32; 2], T, M>;
pub type Chunk3<T, M> = Chunk<[i32; 3], T, M>;

impl<N, T> Chunk<N, T, ()> {
    /// Constructs a chunk without metadata.
    pub fn with_array(array: ArrayN<N, T>) -> Self {
        Chunk {
            metadata: (),
            array,
        }
    }
}

pub struct FastCompressedChunk<N, T, M = ()> {
    pub metadata: M, // metadata doesn't get compressed, hope it's small!
    pub compressed_array: FastLz4CompressedArrayN<N, T>,
}

// PERF: cloning the metadata is unfortunate

impl<N, T, M> Decompressible<FastLz4> for FastCompressedChunk<N, T, M>
where
    T: Copy,
    M: Clone,
    ExtentN<N>: IntegerExtent<N>,
{
    type Decompressed = Chunk<N, T, M>;

    fn decompress(&self) -> Self::Decompressed {
        Chunk {
            metadata: self.metadata.clone(),
            array: self.compressed_array.decompress(),
        }
    }
}

impl<N, T, M> Compressible<FastLz4> for Chunk<N, T, M>
where
    T: Copy,
    M: Clone,
    ExtentN<N>: IntegerExtent<N>,
{
    type Compressed = FastCompressedChunk<N, T, M>;

    fn compress(&self, params: FastLz4) -> Self::Compressed {
        FastCompressedChunk {
            metadata: self.metadata.clone(),
            compressed_array: self.array.compress(params),
        }
    }
}

pub trait ChunkShape<N> {
    /// Makes the mask required to convert points to chunk keys.
    fn mask(&self) -> PointN<N>;

    /// A chunk key is just the leading m bits of each component of a point, where m depends on the
    /// size of the chunk. It can also be interpreted as the minimum point of a chunk extent.
    fn chunk_key_containing_point(mask: &PointN<N>, p: &PointN<N>) -> PointN<N>;
}

impl ChunkShape<[i32; 2]> for Point2i {
    fn mask(&self) -> Point2i {
        assert!(self.dimensions_are_powers_of_2());

        PointN([!(self.x() - 1), !(self.y() - 1)])
    }

    fn chunk_key_containing_point(mask: &Point2i, p: &Point2i) -> Point2i {
        PointN([mask.x() & p.x(), mask.y() & p.y()])
    }
}

impl ChunkShape<[i32; 3]> for Point3i {
    fn mask(&self) -> Point3i {
        assert!(self.dimensions_are_powers_of_2());

        PointN([!(self.x() - 1), !(self.y() - 1), !(self.z() - 1)])
    }

    fn chunk_key_containing_point(mask: &Point3i, p: &Point3i) -> Point3i {
        PointN([mask.x() & p.x(), mask.y() & p.y(), mask.z() & p.z()])
    }
}

impl<N, T, M> ChunkMap<N, T, M>
where
    T: Copy,
    M: Clone,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    /// Creates an empty map.
    pub fn new(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: M,
        compression_params: FastLz4,
    ) -> Self {
        Self {
            chunk_shape,
            chunk_shape_mask: chunk_shape.mask(),
            ambient_value,
            default_chunk_metadata,
            chunks: CompressibleFnvMap::new(compression_params),
        }
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
    pub fn chunk_key(&self, point: &PointN<N>) -> PointN<N> {
        PointN::chunk_key_containing_point(self.chunk_shape_mask(), point)
    }

    /// Returns an iterator over all chunk keys for chunks that overlap the given extent.
    pub fn chunk_keys_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        chunk_keys_for_extent(self.chunk_shape, extent)
    }

    /// The extent spanned by the chunk at `key`.
    pub fn extent_for_chunk_at_key(&self, key: &PointN<N>) -> ExtentN<N> {
        extent_for_chunk_at_key(&self.chunk_shape, key)
    }

    /// Returns the chunk at `key` if it exists.
    pub fn get_chunk<'a>(
        &'a self,
        key: PointN<N>,
        local_cache: &'a LocalChunkCache<N, T, M>,
    ) -> Option<&Chunk<N, T, M>> {
        self.chunks.get_const(key, local_cache)
    }

    /// Returns the mutable chunk at `key` if it exists.
    pub fn get_mut_chunk(&mut self, key: PointN<N>) -> Option<&mut Chunk<N, T, M>> {
        self.chunks.get_mut(key)
    }

    /// Get mutable chunk for `key`. If `key` does not exist, calls `fill_empty_chunk` to fill that
    /// entry first.
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl Fn(&PointN<N>, &ExtentN<N>) -> Chunk<N, T, M>,
    ) -> &mut Chunk<N, T, M> {
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
    #[allow(clippy::type_complexity)]
    pub fn get_chunk_containing_point<'a>(
        &'a self,
        point: &PointN<N>,
        local_cache: &'a LocalChunkCache<N, T, M>,
    ) -> Option<(PointN<N>, &Chunk<N, T, M>)> {
        let chunk_key = self.chunk_key(point);

        self.get_chunk(chunk_key, local_cache)
            .map(|c| (chunk_key, c))
    }

    /// Returns the mutable chunk containing `point` if it exists.
    #[allow(clippy::type_complexity)]
    pub fn get_mut_chunk_containing_point(
        &mut self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &mut Chunk<N, T, M>)> {
        let chunk_key = self.chunk_key(point);

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
        let key = self.chunk_key(p);
        let chunk = self.get_mut_chunk_or_insert_with(key, create_chunk);

        (key, chunk.array.get_unchecked_mut_release(p))
    }

    /// Sets point `p` to value `T`. If `p` is in a chunk that doesn't exist yet, then the chunk
    /// will first be filled with the ambient value and default metadata.
    pub fn get_mut_and_key(&mut self, p: &PointN<N>) -> (PointN<N>, &mut T)
    where
        ArrayN<N, T>: Array<N>,
    {
        let key = self.chunk_key(p);
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

    /// Compressed the least-recently-used chunk using LZ4 compression. On access, compressed chunks
    /// will be decompressed and cached.
    pub fn compress_lru_chunk(&mut self) {
        self.chunks.compress_lru();
    }

    /// Consumes and flushes the chunk cache into the chunk map. This is not strictly necessary, but
    /// it will help with caching efficiency.
    pub fn flush_chunk_cache(&mut self, local_cache: LocalChunkCache<N, T, M>) {
        self.chunks.flush_local_cache(local_cache);
    }

    /// Returns a serializable version of this map. This will compress every chunk in a portable
    /// way.
    pub fn to_serializable(&self, params: BincodeLz4) -> SerializableChunkMap<N, T, M>
    where
        Chunk<N, T, M>: DeserializeOwned + Serialize,
    {
        // PERF: this could easily be done in parallel
        let portable_chunks = self
            .chunks
            .iter_maybe_compressed()
            .map(|(chunk_key, chunk)| {
                let portable_chunk: BincodeLz4Compressed<Chunk<N, T, M>> = match chunk {
                    MaybeCompressed::Compressed(compressed_chunk) => {
                        compressed_chunk.decompress().compress(params)
                    }
                    MaybeCompressed::Decompressed(chunk) => chunk.compress(params),
                };

                (*chunk_key, portable_chunk)
            })
            .collect();

        SerializableChunkMap {
            chunk_shape: self.chunk_shape,
            ambient_value: self.ambient_value,
            default_chunk_metadata: self.default_chunk_metadata.clone(),
            compressed_chunks: portable_chunks,
        }
    }

    /// Returns a new map from the serialized, compressed version. This will decompress each chunk
    /// and compress it again, but in a faster format.
    pub fn from_serializable(map: &SerializableChunkMap<N, T, M>, params: FastLz4) -> Self
    where
        Chunk<N, T, M>: DeserializeOwned + Serialize,
    {
        // PERF: this could easily be done in parallel
        let mut compressible_map = CompressibleFnvMap::new(params);
        for (chunk_key, compressed_chunk) in map.compressed_chunks.iter() {
            compressible_map.insert(*chunk_key, compressed_chunk.decompress());
            compressible_map.compress_lru();
        }

        Self {
            chunk_shape: map.chunk_shape,
            chunk_shape_mask: map.chunk_shape.mask(),
            ambient_value: map.ambient_value,
            default_chunk_metadata: map.default_chunk_metadata.clone(),
            chunks: compressible_map,
        }
    }
}

impl<N, T, M> ChunkMap<N, T, M>
where
    Self: ForEachMut<N, PointN<N>, Data = T>,
    T: Copy,
    M: Clone,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T) {
        self.for_each_mut(extent, |_p, v| *v = value);
    }
}

/// A thread-local reader of a `ChunkMap` which stores a cache of chunks that were
/// decompressed after missing the global cache of chunks.
pub struct ChunkMapReader<'a, N, T, M = ()>
where
    T: Copy,
    M: Clone,
    PointN<N>: Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    map: &'a ChunkMap<N, T, M>,
    local_cache: &'a LocalChunkCache<N, T, M>,
}

pub type ChunkMapReader2<'a, T, M> = ChunkMapReader<'a, [i32; 2], T, M>;
pub type ChunkMapReader3<'a, T, M> = ChunkMapReader<'a, [i32; 3], T, M>;

impl<'a, N, T, M> ChunkMapReader<'a, N, T, M>
where
    T: Copy,
    M: Clone,
    PointN<N>: Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    /// Construct a new reader for `map` using a `local_cache`.
    pub fn new(map: &'a ChunkMap<N, T, M>, local_cache: &'a LocalChunkCache<N, T, M>) -> Self {
        Self { map, local_cache }
    }
}

/// Call `ChunkMap::to_serializable` to get this type, which is an LZ4-compressed,
/// serde-serializable type.
#[allow(clippy::type_complexity)]
#[derive(Deserialize, Serialize)]
pub struct SerializableChunkMap<N, T, M = ()>
where
    PointN<N>: Eq + Hash,
{
    pub chunk_shape: PointN<N>,
    pub ambient_value: T,
    pub default_chunk_metadata: M,
    pub compressed_chunks: FnvHashMap<PointN<N>, BincodeLz4Compressed<Chunk<N, T, M>>>,
}

pub type SerializableChunkMap2<T, M> = SerializableChunkMap<[i32; 2], T, M>;
pub type SerializableChunkMap3<T, M> = SerializableChunkMap<[i32; 3], T, M>;

/// An extent that takes the same value everywhere.
#[derive(Copy, Clone)]
pub struct AmbientExtent<N, T> {
    pub value: T,
    _n: std::marker::PhantomData<N>,
}

pub type AmbientExtent2<T> = AmbientExtent<[i32; 2], T>;
pub type AmbientExtent3<T> = AmbientExtent<[i32; 3], T>;

impl<N, T> AmbientExtent<N, T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _n: Default::default(),
        }
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.value.clone()
    }
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
    chunk_shape: PointN<N>,
    extent: &ExtentN<N>,
) -> impl Iterator<Item = PointN<N>>
where
    PointN<N>: Point + Ones,
    ExtentN<N>: IntegerExtent<N>,
{
    let key_min = extent.minimum / chunk_shape;
    let key_max = extent.max() / chunk_shape;

    ExtentN::from_min_and_max(key_min, key_max)
        .iter_points()
        .map(move |p| p * chunk_shape)
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, M> GetMut<&PointN<N>> for ChunkMap<N, T, M>
where
    T: Copy,
    M: Clone,
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

impl<'a, N, T, M> GetRef<&PointN<N>> for ChunkMapReader<'a, N, T, M>
where
    T: Copy,
    M: Clone,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N>,
{
    type Data = T;

    fn get_ref(&self, p: &PointN<N>) -> &Self::Data {
        self.map
            .get_chunk_containing_point(p, &self.local_cache)
            .map(|(_key, chunk)| chunk.array.get_unchecked_ref_release(p))
            .unwrap_or(&self.map.ambient_value)
    }
}

// TODO: could be more generic once Rust has specialization
impl<'a, N, T, M> Get<&PointN<N>> for ChunkMapReader<'a, N, T, M>
where
    Self: for<'b> GetRef<&'b PointN<N>, Data = T>,
    T: Copy,
    M: Clone,
    PointN<N>: Point + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    type Data = T;

    fn get(&self, p: &PointN<N>) -> Self::Data {
        *self.get_ref(p)
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<'a, N, T, M> ForEachRef<N, PointN<N>> for ChunkMapReader<'a, N, T, M>
where
    T: Copy,
    M: Clone,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N> + ForEachRef<N, PointN<N>, Data = T>,
{
    type Data = T;

    fn for_each_ref(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &Self::Data)) {
        for chunk_key in self.map.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.map.get_chunk(chunk_key, &self.local_cache) {
                chunk.array.for_each_ref(extent, |p, value| f(p, value));
            } else {
                let chunk_extent = self.map.extent_for_chunk_at_key(&chunk_key);
                AmbientExtent::new(self.map.ambient_value)
                    .for_each_ref(&extent.intersection(&chunk_extent), |p, value| f(p, value))
            }
        }
    }
}

impl<N, T> ForEachRef<N, PointN<N>> for AmbientExtent<N, T>
where
    ExtentN<N>: IntegerExtent<N>,
{
    type Data = T;

    fn for_each_ref(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &Self::Data)) {
        for p in extent.iter_points() {
            f(p, &self.value);
        }
    }
}

impl<'a, N, T, M> ForEachMut<N, PointN<N>> for ChunkMap<N, T, M>
where
    T: Copy,
    M: Clone,
    PointN<N>: Point + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: ForEachMut<N, PointN<N>, Data = T>,
{
    type Data = T;

    fn for_each_mut(&mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &mut Self::Data)) {
        let ChunkMap {
            chunk_shape,
            ambient_value,
            default_chunk_metadata,
            chunks,
            ..
        } = self;

        for chunk_key in chunk_keys_for_extent(*chunk_shape, extent) {
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

impl<'a, N, T, M> ReadExtent<'a, N> for ChunkMapReader<'a, N, T, M>
where
    T: Copy,
    M: Clone,
    ArrayN<N, T>: Array<N>,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    type Src = ArrayChunkCopySrc<'a, N, T>;
    type SrcIter = ArrayChunkCopySrcIter<'a, N, T>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .map
            .chunk_keys_for_extent(extent)
            .map(|key| {
                let chunk_extent = self.map.extent_for_chunk_at_key(&key);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.map
                        .get_chunk(key, &self.local_cache)
                        .map(|chunk| Either::Left(ArrayCopySrc(&chunk.array)))
                        .unwrap_or_else(|| {
                            Either::Right(AmbientExtent::new(self.map.ambient_value))
                        }),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

pub type ChunkCopySrcIter<M, N, T> = std::vec::IntoIter<(ExtentN<N>, ChunkCopySrc<M, N, T>)>;
pub type ChunkCopySrc<M, N, T> = Either<ArrayCopySrc<M>, AmbientExtent<N, T>>;

pub type ArrayChunkCopySrcIter<'a, N, T> =
    std::vec::IntoIter<(ExtentN<N>, ArrayChunkCopySrc<'a, N, T>)>;
pub type ArrayChunkCopySrc<'a, N, T> = Either<ArrayCopySrc<&'a ArrayN<N, T>>, AmbientExtent<N, T>>;

// If ArrayN supports writing from type Src, then so does ChunkMap.
impl<'a, N, T, M, Src> WriteExtent<N, Src> for ChunkMap<N, T, M>
where
    T: Copy,
    M: Clone,
    Src: Copy,
    PointN<N>: Point + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: WriteExtent<N, Src>,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        let ChunkMap {
            chunk_shape,
            ambient_value,
            default_chunk_metadata,
            chunks,
            ..
        } = self;

        for chunk_key in chunk_keys_for_extent(*chunk_shape, extent) {
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{copy_extent, Array3};

    use building_blocks_core::Extent3i;

    #[test]
    fn chunk_keys_for_extent_gives_keys_for_chunks_overlapping_extent() {
        let chunk_shape = PointN([16; 3]);
        let query_extent = Extent3i::from_min_and_shape(PointN([15; 3]), PointN([16; 3]));
        let chunk_keys: Vec<_> = chunk_keys_for_extent(chunk_shape, &query_extent).collect();

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
        let mut map = ChunkMap3::new(chunk_shape, ambient_value, (), FastLz4 { level: 10 });

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
        let mut map = ChunkMap3::new(chunk_shape, ambient_value, (), FastLz4 { level: 10 });

        let write_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        map.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        let local_cache = LocalChunkCache3::new();
        let reader = ChunkMapReader3::new(&map, &local_cache);
        for p in read_extent.iter_points() {
            if write_extent.contains(&p) {
                assert_eq!(reader.get_ref(&p), &1);
            } else {
                assert_eq!(reader.get_ref(&p), &0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        let array = Array3::fill(extent_to_copy, 1);

        let chunk_shape = PointN([16; 3]);
        let ambient_value = 0;
        let mut map = ChunkMap3::new(chunk_shape, ambient_value, (), FastLz4 { level: 10 });

        copy_extent(&extent_to_copy, &array, &mut map);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        let local_cache = LocalChunkCache3::new();
        let reader = ChunkMapReader3::new(&map, &local_cache);
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(&p) {
                assert_eq!(reader.get_ref(&p), &1);
            } else {
                assert_eq!(reader.get_ref(&p), &0);
            }
        }
    }
}
