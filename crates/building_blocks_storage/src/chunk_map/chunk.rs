use crate::array::{ArrayN, FastArrayCompression};

use building_blocks_core::prelude::*;

use compressible_map::{
    BincodeCompression, BytesCompression, Compressed, Compression, MaybeCompressed,
};
use serde::{Deserialize, Serialize};

/// One piece of the `ChunkMap`. Contains both some generic metadata and the data for each point in
/// the chunk extent.
#[derive(Clone, Deserialize, Serialize)]
pub struct Chunk<N, T, M> {
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

#[derive(Copy, Clone)]
pub struct FastChunkCompression<N, T, M, B> {
    pub array_compression: FastArrayCompression<N, T, B>,
    marker: std::marker::PhantomData<(N, T, M)>,
}

impl<N, T, M, B> FastChunkCompression<N, T, M, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            array_compression: FastArrayCompression::new(bytes_compression),
            marker: Default::default(),
        }
    }
}

pub struct FastCompressedChunk<N, T, M, B>
where
    T: Copy,
    B: BytesCompression,
    ExtentN<N>: IntegerExtent<N>,
{
    pub metadata: M, // metadata doesn't get compressed, hope it's small!
    pub compressed_array: Compressed<FastArrayCompression<N, T, B>>,
}

impl<N, T, M, B> Compression for FastChunkCompression<N, T, M, B>
where
    T: Copy,
    M: Clone,
    B: BytesCompression,
    ExtentN<N>: IntegerExtent<N>,
{
    type Data = Chunk<N, T, M>;
    type CompressedData = FastCompressedChunk<N, T, M, B>;

    // PERF: cloning the metadata is unfortunate

    fn compress(&self, chunk: &Self::Data) -> Compressed<Self> {
        Compressed::new(FastCompressedChunk {
            metadata: chunk.metadata.clone(),
            compressed_array: self.array_compression.compress(&chunk.array),
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        Chunk {
            metadata: compressed.metadata.clone(),
            array: compressed.compressed_array.decompress(),
        }
    }
}

pub type BincodeChunkCompression<N, T, M, B> = BincodeCompression<Chunk<N, T, M>, B>;
pub type BincodeCompressedChunk<N, T, M, B> = Compressed<BincodeCompression<Chunk<N, T, M>, B>>;

pub trait ChunkShape<N> {
    /// Makes the mask required to convert points to chunk keys.
    fn mask(&self) -> PointN<N>;

    /// A chunk key is just the leading m bits of each component of a point, where m depends on the
    /// size of the chunk. It can also be interpreted as the minimum point of a chunk extent.
    fn chunk_key_containing_point(mask: &PointN<N>, p: &PointN<N>) -> PointN<N>;

    fn ilog2(&self) -> PointN<N>;
}

macro_rules! impl_chunk_shape {
    ($point:ty, $dims:ty) => {
        impl ChunkShape<$dims> for $point {
            fn mask(&self) -> $point {
                assert!(self.dimensions_are_powers_of_2());

                self.map_components_unary(|c| !(c - 1))
            }

            fn chunk_key_containing_point(mask: &$point, p: &$point) -> $point {
                mask.map_components_binary(p, |c1, c2| c1 & c2)
            }

            fn ilog2(&self) -> $point {
                self.map_components_unary(|c| c.trailing_zeros() as i32)
            }
        }
    };
}

impl_chunk_shape!(Point2i, [i32; 2]);
impl_chunk_shape!(Point3i, [i32; 3]);

pub type MaybeCompressedChunk<N, T, M, B> =
    MaybeCompressed<Chunk<N, T, M>, Compressed<FastChunkCompression<N, T, M, B>>>;
