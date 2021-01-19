//! A sparse lattice map made of up array chunks.
//!
//! # Addressing
//!
//! The data can either be addressed by chunk key with the `get_chunk*` methods or by individual points using the `Get*` and
//! `ForEach*` trait impls. The map of chunks uses `Point3i` keys. The key for a chunk is the minimum point in that chunk, which
//! is always a multiple of the chunk shape. Chunk shape dimensions must be powers of 2, which allows for efficiently
//! calculating a chunk key from any point in the chunk.
//!
//! # Chunk Storage
//!
//! `ChunkMap<N, T, Meta, Store>` depends on a backing chunk storage `Store`, which can implement some of `ChunkReadStorage` or
//! `ChunkWriteStorage`. A storage can be as simple as a `HashMap`, which provides good performance for both iteration and
//! random access. It could also be something more memory efficient like `CompressibleChunkStorage` or
//! `CompressibleChunkStorageReader`, which perform nearly as well but involve some extra management of the cache.
//!
//! # Serialization
//!
//! In order to efficiently serialize a `ChunkMap`, you can first use `SerializableChunkMap::from_chunk_map` to create a compact
//! serializable representation. It will compress the bincode representation of the chunks.
//!
//! # Example `ChunkHashMap` Usage
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let ambient_value = 0;
//! let builder = ChunkMapBuilder {
//!    chunk_shape: PointN([16; 3]), // components must be powers of 2
//!    ambient_value,
//!    default_chunk_metadata: (), // chunk metadata is optional
//! };
//! let mut map = builder.build_with_hash_map_storage();
//!
//! // Although we only write 3 points, 3 whole dense chunks will be inserted.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Even though the map is sparse, we can get the smallest extent that bounds all of the occupied
//! // chunks.
//! let bounding_extent = map.bounding_extent();
//!
//! // Now we can read back the values.
//! map.for_each(&bounding_extent, |p, value| {
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
//! // You can also access individual points like you can with a `ArrayN`. This is
//! // slower than iterating, because it hashes the chunk coordinates for every access.
//! for p in write_points.iter() {
//!     assert_eq!(map.get(p), 1);
//! }
//! assert_eq!(map.get(&PointN([1, 1, 1])), 0);
//!
//! // Sometimes you need to implement very fast algorithms (like kernel-based methods) that do a
//! // lot of random access. In this case it's most efficient to use `Stride`s, but `ChunkMap`
//! // doesn't support random indexing by `Stride`. Instead, assuming that your query spans multiple
//! // chunks, you should copy the extent into a dense map first. (The copy is fast).
//! let query_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([32; 3]));
//! let mut dense_map = Array3::fill(query_extent, ambient_value);
//! copy_extent(&query_extent, &map, &mut dense_map);
//! ```
//!
//! # Example `CompressibleChunkMap` Usage
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! #
//! # let builder = ChunkMapBuilder {
//! #    chunk_shape: PointN([16; 3]), // components must be powers of 2
//! #    ambient_value: 0,
//! #    default_chunk_metadata: (), // chunk metadata is optional
//! # };
//! #
//! let mut map = builder.build_with_write_storage(CompressibleChunkStorage::new(Lz4 { level: 10 }));
//!
//! // You can write voxels the same as any other `ChunkMap`. As chunks are created, they will be placed in an LRU cache.
//! let write_points = [PointN([-100; 3]), PointN([0; 3]), PointN([100; 3])];
//! for p in write_points.iter() {
//!     *map.get_mut(&p) = 1;
//! }
//!
//! // Save some space by compressing the least recently used chunks. On further access to the compressed chunks, they will get
//! // decompressed and cached.
//! map.storage_mut().compress_lru();
//!
//! // In order to use the read-only access traits, you need to construct a `CompressibleChunkStorageReader`.
//! let local_cache = LocalChunkCache::new();
//! let reader = map.reader(&local_cache);
//!
//! let bounding_extent = reader.bounding_extent();
//! reader.for_each(&bounding_extent, |p, value| {
//!     if write_points.iter().position(|pw| p == *pw) != None {
//!         assert_eq!(value, 1);
//!     } else {
//!         assert_eq!(value, 0);
//!     }
//! });
//!
//! // For efficient caching, you should flush your local cache back into the main storage when you are done with it.
//! map.storage_mut().flush_local_cache(local_cache);
//! ```

mod ambient;
mod chunk;
mod storage;

pub use ambient::*;
pub use chunk::*;
pub use storage::*;

use crate::{
    array::{ArrayCopySrc, ArrayIndexer, ArrayN},
    ForEach, ForEachMut, ForEachRef, Get, GetMut, GetRef, GetUnchecked, GetUncheckedMutRelease,
    GetUncheckedRef, GetUncheckedRefRelease, ReadExtent, WriteExtent,
};

use building_blocks_core::{bounding_extent, ExtentN, IntegerPoint, PointN};

use core::hash::Hash;
use either::Either;
use fnv::FnvHashMap;
use serde::{Deserialize, Serialize};

/// A lattice map made up of same-shaped `ArrayN` chunks. It takes a value at every possible `PointN`, because accesses made
/// outside of the stored chunks will return some ambient value specified on creation.
///
/// `ChunkMap` is generic over the type used to actually store the `Chunk`s. You can use any storage that implements
/// `ChunkReadStorage` or `ChunkWriteStorage`. Being a lattice map, `ChunkMap` will implement various access traits, depending
/// on the capabilities of the chunk storage.
///
/// If the chunk storage implements `ChunkReadStorage`, then `ChunkMap` will implement:
/// - `Get`
/// - `ForEach`
/// - `ReadExtent`
///
/// If the chunk storage implements `ChunkWriteStorage`, then `ChunkMap` will implement:
/// - `GetMut`
/// - `ForEachMut`
/// - `WriteExtent`
pub struct ChunkMap<N, T, Meta, Store> {
    /// Translates from lattice coordinates to chunk key space.
    pub indexer: ChunkIndexer<N>,

    ambient_value: T,
    default_chunk_metadata: Meta,
    storage: Store,
}

/// A 2-dimensional `ChunkMap`.
pub type ChunkMap2<T, Meta, Store> = ChunkMap<[i32; 2], T, Meta, Store>;
/// A 3-dimensional `ChunkMap`.
pub type ChunkMap3<T, Meta, Store> = ChunkMap<[i32; 3], T, Meta, Store>;

/// A few pieces of info used within the `ChunkMap`. You will probably keep one of these around to create new `ChunkMap`s from
/// a chunk storage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ChunkMapBuilder<N, T, Meta = ()> {
    /// The shape of each chunk. All dimensions must be powers of 2.
    pub chunk_shape: PointN<N>,
    /// The value to use when none is specified, i.e. when creating new chunks or accessing vacant chunks.
    pub ambient_value: T,
    /// The metadata value used to initialize new chunks.
    pub default_chunk_metadata: Meta,
}

/// A 2-dimensional `ChunkMapBuilder`.
pub type ChunkMapBuilder2<T, Meta = ()> = ChunkMapBuilder<[i32; 2], T, Meta>;
/// A 3-dimensional `ChunkMapBuilder`.
pub type ChunkMapBuilder3<T, Meta = ()> = ChunkMapBuilder<[i32; 3], T, Meta>;

impl<N, T, Meta> ChunkMapBuilder<N, T, Meta>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Create a new `ChunkMap` with the given `storage` which must implement both `ChunkReadStorage` and `ChunkWriteStorage`.
    pub fn build_with_rw_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Meta, Store>
    where
        Store: ChunkReadStorage<N, T, Meta> + ChunkWriteStorage<N, T, Meta>,
    {
        self.build_with_storage(storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkReadStorage`.
    pub fn build_with_read_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Meta, Store>
    where
        Store: ChunkReadStorage<N, T, Meta>,
    {
        self.build_with_storage(storage)
    }

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkWriteStorage`.
    pub fn build_with_write_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Meta, Store>
    where
        Store: ChunkWriteStorage<N, T, Meta>,
    {
        self.build_with_storage(storage)
    }

    /// Create a new `ChunkMap` using a `FnvHashMap` as the chunk storage.
    pub fn build_with_hash_map_storage(self) -> ChunkHashMap<N, T, Meta>
    where
        PointN<N>: Hash,
    {
        self.build_with_rw_storage(FnvHashMap::default())
    }

    fn build_with_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Meta, Store> {
        ChunkMap::new(
            self.chunk_shape,
            self.ambient_value,
            self.default_chunk_metadata,
            storage,
        )
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store> {
    /// Consumes `self` and returns the backing chunk storage.
    #[inline]
    pub fn take_storage(self) -> Store {
        self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage(&self) -> &Store {
        &self.storage
    }

    /// Borrows the internal chunk storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut Store {
        &mut self.storage
    }

    /// Return the ambient value of the map.
    #[inline]
    pub fn ambient_value(&self) -> T
    where
        T: Copy,
    {
        self.ambient_value
    }

    /// Return the default metadata for chunks.
    #[inline]
    pub fn default_chunk_metadata(&self) -> &Meta {
        &self.default_chunk_metadata
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Creates a map using the given `storage`.
    ///
    /// All dimensions of `chunk_shape` must be powers of 2.
    fn new(
        chunk_shape: PointN<N>,
        ambient_value: T,
        default_chunk_metadata: Meta,
        storage: Store,
    ) -> Self {
        Self {
            indexer: ChunkIndexer::new(chunk_shape),
            ambient_value,
            default_chunk_metadata,
            storage,
        }
    }

    /// Get the `ChunkMapBuilder` used to build this map.
    pub fn builder(&self) -> ChunkMapBuilder<N, T, Meta>
    where
        T: Copy,
        Meta: Clone,
    {
        ChunkMapBuilder {
            chunk_shape: self.indexer.chunk_shape(),
            ambient_value: self.ambient_value,
            default_chunk_metadata: self.default_chunk_metadata.clone(),
        }
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: ChunkReadStorage<N, T, Meta>,
{
    /// Borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_chunk(&self, key: &PointN<N>) -> Option<&Chunk<N, T, Meta>> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get(key)
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be represented by an `AmbientExtent`.
    #[inline]
    pub fn visit_chunks(
        &self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(Either<&Chunk<N, T, Meta>, (&ExtentN<N>, AmbientExtent<N, T>)>),
    ) where
        T: Copy,
    {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(&chunk_key) {
                visitor(Either::Left(chunk))
            } else {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(chunk_key);
                visitor(Either::Right((
                    &chunk_extent,
                    AmbientExtent::new(self.ambient_value),
                )))
            }
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_chunks(
        &self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&Chunk<N, T, Meta>),
    ) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_chunk(&chunk_key) {
                visitor(chunk)
            }
        }
    }
}

impl<N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    /// Overwrite the `Chunk` at `key` with `chunk`. Drops the previous value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn write_chunk(&mut self, key: PointN<N>, chunk: Chunk<N, T, Meta>) {
        debug_assert!(chunk.array.extent().shape.eq(&self.indexer.chunk_shape()));
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.write(key, chunk);
    }

    /// Replace the `Chunk` at `key` with `chunk`, returning the old value.
    ///
    /// In debug mode only, asserts that `key` is valid and `chunk`'s shape is valid.
    #[inline]
    pub fn replace_chunk(
        &mut self,
        key: PointN<N>,
        chunk: Chunk<N, T, Meta>,
    ) -> Option<Chunk<N, T, Meta>> {
        debug_assert!(chunk.array.extent().shape.eq(&self.indexer.chunk_shape()));
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.replace(key, chunk)
    }

    /// Mutably borrow the chunk at `key`.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk(&mut self, key: &PointN<N>) -> Option<&mut Chunk<N, T, Meta>> {
        debug_assert!(self.indexer.chunk_key_is_valid(key));

        self.storage.get_mut(key)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, `create_chunk` is called to insert one.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_with(
        &mut self,
        key: PointN<N>,
        create_chunk: impl FnOnce() -> Chunk<N, T, Meta>,
    ) -> &mut Chunk<N, T, Meta> {
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        self.storage.get_mut_or_insert_with(key, create_chunk)
    }

    /// Mutably borrow the chunk at `key`. If the chunk doesn't exist, a new chunk is created with the ambient value.
    ///
    /// In debug mode only, asserts that `key` is valid.
    #[inline]
    pub fn get_mut_chunk_or_insert_ambient(&mut self, key: PointN<N>) -> &mut Chunk<N, T, Meta>
    where
        T: Copy,
        Meta: Clone,
    {
        debug_assert!(self.indexer.chunk_key_is_valid(&key));

        let Self {
            indexer,
            ambient_value,
            default_chunk_metadata,
            storage,
            ..
        } = self;

        storage.get_mut_or_insert_with(key, || Chunk {
            metadata: default_chunk_metadata.clone(),
            array: ArrayN::fill(indexer.extent_for_chunk_at_key(key), *ambient_value),
        })
    }

    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be created first with ambient value.
    #[inline]
    pub fn visit_mut_chunks(
        &mut self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Chunk<N, T, Meta>),
    ) where
        T: Copy,
        Meta: Clone,
    {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            visitor(self.get_mut_chunk_or_insert_ambient(chunk_key));
        }
    }

    /// Call `visitor` on all occupied chunks that overlap `extent`.
    #[inline]
    pub fn visit_occupied_mut_chunks(
        &mut self,
        extent: &ExtentN<N>,
        mut visitor: impl FnMut(&mut Chunk<N, T, Meta>),
    ) {
        for chunk_key in self.indexer.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.get_mut_chunk(&chunk_key) {
                visitor(chunk)
            }
        }
    }

    /// Fill all of `extent` with the same `value`.
    #[inline]
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T)
    where
        Self: ForEachMut<N, PointN<N>, Data = T>,
        T: Copy,
    {
        self.for_each_mut(extent, |_p, v| *v = value);
    }
}

impl<'a, N, T, Meta, Store> ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: IterChunkKeys<'a, N>,
{
    /// The smallest extent that bounds all chunks.
    pub fn bounding_extent(&'a self) -> ExtentN<N> {
        bounding_extent(self.storage.chunk_keys().flat_map(|key| {
            let chunk_extent = self.indexer.extent_for_chunk_at_key(*key);

            vec![chunk_extent.minimum, chunk_extent.max()].into_iter()
        }))
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, Meta, Store> GetRef<&PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    N: ArrayIndexer<N>,
    Store: ChunkReadStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn get_ref(&self, p: &PointN<N>) -> &Self::Data {
        let key = self.indexer.chunk_key_containing_point(p);

        self.get_chunk(&key)
            .map(|chunk| chunk.array.get_unchecked_ref_release(p))
            .unwrap_or(&self.ambient_value)
    }
}

impl<N, T, Meta, Store> GetMut<&PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    N: ArrayIndexer<N>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, p: &PointN<N>) -> &mut Self::Data {
        let key = self.indexer.chunk_key_containing_point(p);
        let chunk = self.get_mut_chunk_or_insert_ambient(key);

        chunk.array.get_unchecked_mut_release(p)
    }
}

impl_get_via_get_ref_and_clone!(ChunkMap<N, T, Meta, Store>, N, T, Meta, Store);

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<N, T, Meta, Store> ForEach<N, PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    Self: ForEachRef<N, PointN<N>>,
    <Self as ForEachRef<N, PointN<N>>>::Data: Clone,
{
    type Data = <Self as ForEachRef<N, PointN<N>>>::Data;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Data)) {
        self.for_each_ref(extent, |p, data| f(p, data.clone()));
    }
}

impl<N, T, Meta, Store> ForEachRef<N, PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    ArrayN<N, T>: ForEachRef<N, PointN<N>, Data = T>,
    T: Copy,
    Store: ChunkReadStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn for_each_ref(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &Self::Data)) {
        self.visit_chunks(extent, |chunk| match chunk {
            Either::Left(chunk) => {
                chunk.array.for_each_ref(extent, |p, value| f(p, value));
            }
            Either::Right((chunk_extent, ambient)) => {
                ambient.for_each(&extent.intersection(&chunk_extent), |p, value| f(p, &value))
            }
        });
    }
}

impl<N, T, Meta, Store> ForEachMut<N, PointN<N>> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    ArrayN<N, T>: ForEachMut<N, PointN<N>, Data = T>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
{
    type Data = T;

    #[inline]
    fn for_each_mut(&mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, &mut Self::Data)) {
        self.visit_mut_chunks(extent, |chunk| {
            chunk.array.for_each_mut(extent, |p, value| f(p, value))
        });
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, N, T, Meta, Store> ReadExtent<'a, N> for ChunkMap<N, T, Meta, Store>
where
    N: ArrayIndexer<N>,
    PointN<N>: 'a + IntegerPoint<N>,
    T: 'a + Copy,
    Store: ChunkReadStorage<N, T, Meta>,
{
    type Src = ArrayChunkCopySrc<'a, N, T>;
    type SrcIter = ArrayChunkCopySrcIter<'a, N, T>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .indexer
            .chunk_keys_for_extent(extent)
            .map(|key| {
                let chunk_extent = self.indexer.extent_for_chunk_at_key(key);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.get_chunk(&key)
                        .map(|chunk| Either::Left(ArrayCopySrc(&chunk.array)))
                        .unwrap_or_else(|| Either::Right(AmbientExtent::new(self.ambient_value))),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If ArrayN supports writing from type Src, then so does ChunkMap.
impl<N, T, Meta, Store, Src> WriteExtent<N, Src> for ChunkMap<N, T, Meta, Store>
where
    PointN<N>: IntegerPoint<N>,
    ArrayN<N, T>: WriteExtent<N, Src>,
    T: Copy,
    Meta: Clone,
    Store: ChunkWriteStorage<N, T, Meta>,
    Src: Copy,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        self.visit_mut_chunks(extent, |chunk| chunk.array.write_extent(extent, src));
    }
}

#[doc(hidden)]
pub type ChunkCopySrc<Map, N, T> = Either<ArrayCopySrc<Map>, AmbientExtent<N, T>>;
#[doc(hidden)]
pub type ArrayChunkCopySrcIter<'a, N, T> =
    std::vec::IntoIter<(ExtentN<N>, ArrayChunkCopySrc<'a, N, T>)>;
#[doc(hidden)]
pub type ArrayChunkCopySrc<'a, N, T> = Either<ArrayCopySrc<&'a ArrayN<N, T>>, AmbientExtent<N, T>>;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{access::Get, copy_extent, Array3};

    use building_blocks_core::Extent3i;

    const BUILDER: ChunkMapBuilder3<i32> = ChunkMapBuilder {
        chunk_shape: PointN([16; 3]),
        ambient_value: 0,
        default_chunk_metadata: (),
    };

    #[test]
    fn write_and_read_points() {
        let mut map = BUILDER.build_with_hash_map_storage();

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
        let mut map = BUILDER.build_with_hash_map_storage();

        let write_extent = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        map.for_each_mut(&write_extent, |_p, value| *value = 1);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        for p in read_extent.iter_points() {
            if write_extent.contains(&p) {
                assert_eq!(map.get(&p), 1);
            } else {
                assert_eq!(map.get(&p), 0);
            }
        }
    }

    #[test]
    fn copy_extent_from_array_then_read() {
        let extent_to_copy = Extent3i::from_min_and_shape(PointN([10; 3]), PointN([80; 3]));
        let array = Array3::fill(extent_to_copy, 1);

        let mut map = BUILDER.build_with_hash_map_storage();

        copy_extent(&extent_to_copy, &array, &mut map);

        let read_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
        for p in read_extent.iter_points() {
            if extent_to_copy.contains(&p) {
                assert_eq!(map.get(&p), 1);
            } else {
                assert_eq!(map.get(&p), 0);
            }
        }
    }
}
