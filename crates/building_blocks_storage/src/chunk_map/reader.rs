use super::{Chunk, ChunkMap, ChunkShape, LocalChunkCache};

use crate::{
    access::{ForEach, GetUncheckedRelease, ReadExtent},
    array::{Array, ArrayCopySrc, ArrayN},
    Get,
};

use building_blocks_core::{ExtentN, IntegerExtent, IntegerPoint, PointN};

use compressible_map::{BytesCompression, Lz4};
use core::hash::Hash;
use either::Either;

/// A thread-local reader of a `ChunkMap` which stores a cache of chunks that were
/// decompressed after missing the global cache of chunks.
pub struct ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    pub map: &'a ChunkMap<N, T, M, B>,
    pub local_cache: &'a LocalChunkCache<N, T, M>,
}

pub type ChunkMapReader2<'a, T, M = (), B = Lz4> = ChunkMapReader<'a, [i32; 2], T, M, B>;
pub type ChunkMapReader3<'a, T, M = (), B = Lz4> = ChunkMapReader<'a, [i32; 3], T, M, B>;

impl<'a, N, T, M, B> ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: ChunkShape<N> + Eq + Hash + IntegerPoint,
    ExtentN<N>: IntegerExtent<N>,
{
    /// Construct a new reader for `map` using a `local_cache`.
    pub fn new(map: &'a ChunkMap<N, T, M, B>, local_cache: &'a LocalChunkCache<N, T, M>) -> Self {
        Self { map, local_cache }
    }

    pub fn get_chunk_containing_point(
        &self,
        point: &PointN<N>,
    ) -> Option<(PointN<N>, &Chunk<N, T, M>)> {
        self.map
            .get_chunk_containing_point(point, &self.local_cache)
    }

    pub fn get_chunk(&self, key: PointN<N>) -> Option<&Chunk<N, T, M>> {
        self.map.get_chunk(key, &self.local_cache)
    }
}

impl<'a, N, T, M, B> std::ops::Deref for ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
{
    type Target = ChunkMap<N, T, M, B>;

    fn deref(&self) -> &Self::Target {
        self.map
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, N, T, M, B> Get<&PointN<N>> for ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N>,
{
    type Data = T;

    fn get(&self, p: &PointN<N>) -> Self::Data {
        self.map
            .get_chunk_containing_point(p, &self.local_cache)
            .map(|(_key, chunk)| chunk.array.get_unchecked_release(p))
            .unwrap_or(self.map.ambient_value)
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<'a, N, T, M, B> ForEach<N, PointN<N>> for ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint + ChunkShape<N> + Eq + Hash,
    ExtentN<N>: IntegerExtent<N>,
    ArrayN<N, T>: Array<N> + ForEach<N, PointN<N>, Data = T>,
{
    type Data = T;

    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Data)) {
        for chunk_key in self.map.chunk_keys_for_extent(extent) {
            if let Some(chunk) = self.map.get_chunk(chunk_key, &self.local_cache) {
                chunk.array.for_each(extent, |p, value| f(p, value));
            } else {
                let chunk_extent = self.map.extent_for_chunk_at_key(&chunk_key);
                AmbientExtent::new(self.map.ambient_value)
                    .for_each(&extent.intersection(&chunk_extent), |p, value| f(p, value))
            }
        }
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, N, T, M, B> ReadExtent<'a, N> for ChunkMapReader<'a, N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
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

pub type ChunkCopySrc<M, N, T> = Either<ArrayCopySrc<M>, AmbientExtent<N, T>>;

pub type ArrayChunkCopySrcIter<'a, N, T> =
    std::vec::IntoIter<(ExtentN<N>, ArrayChunkCopySrc<'a, N, T>)>;
pub type ArrayChunkCopySrc<'a, N, T> = Either<ArrayCopySrc<&'a ArrayN<N, T>>, AmbientExtent<N, T>>;

/// An extent that takes the same value everywhere.
#[derive(Copy, Clone)]
pub struct AmbientExtent<N, T> {
    pub value: T,
    _n: std::marker::PhantomData<N>,
}

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

impl<N, T> ForEach<N, PointN<N>> for AmbientExtent<N, T>
where
    T: Clone,
    ExtentN<N>: IntegerExtent<N>,
{
    type Data = T;

    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Data)) {
        for p in extent.iter_points() {
            f(p, self.value.clone());
        }
    }
}
