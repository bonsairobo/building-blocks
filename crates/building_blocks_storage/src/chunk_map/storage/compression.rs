use super::Chunk;

use crate::{
    array::FastArrayCompression,
    compression::{BincodeCompression, BytesCompression, Compressed, Compression, MaybeCompressed},
};

use building_blocks_core::prelude::*;

/// A `Compression` used for compressing `Chunk`s. It just uses the internal `FastArrayCompression` and clones the metadata.
#[derive(Copy, Clone)]
pub struct FastChunkCompression<N, T, M, B> {
    pub array_compression: FastArrayCompression<N, T, B>,
    marker: std::marker::PhantomData<M>,
}

impl<N, T, M, B> FastChunkCompression<N, T, M, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            array_compression: FastArrayCompression::new(bytes_compression),
            marker: Default::default(),
        }
    }
}

/// The target of `FastChunkCompression`. You probably want to use `Compressed<FastChunkCompression>` instead.
#[derive(Clone)]
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

pub type MaybeCompressedChunk<N, T, M, B> =
    MaybeCompressed<Chunk<N, T, M>, Compressed<FastChunkCompression<N, T, M, B>>>;

pub type MaybeCompressedChunkRef<'a, N, T, M, B> =
    MaybeCompressed<&'a Chunk<N, T, M>, &'a Compressed<FastChunkCompression<N, T, M, B>>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use super::*;
        use crate::$backend;

        pub type MaybeCompressedChunk2<T, M = (), B = $backend> =
            MaybeCompressedChunk<[i32; 2], T, M, B>;
        pub type MaybeCompressedChunk3<T, M = (), B = $backend> =
            MaybeCompressedChunk<[i32; 3], T, M, B>;
        pub type MaybeCompressedChunkRef2<'a, T, M = (), B = $backend> =
            MaybeCompressedChunkRef<'a, [i32; 2], T, M, B>;
        pub type MaybeCompressedChunkRef3<'a, T, M = (), B = $backend> =
            MaybeCompressedChunkRef<'a, [i32; 3], T, M, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if we only use one, then we want to have these aliases refer to the choice we
// made.
pub mod conditional_aliases {
    #[cfg(all(feature = "lz4", not(feature = "snap")))]
    define_conditional_aliases!(Lz4);
    #[cfg(all(not(feature = "lz4"), feature = "snap"))]
    define_conditional_aliases!(Snappy);
}
