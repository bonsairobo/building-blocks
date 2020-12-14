use super::Chunk;

use crate::{
    array::FastArrayCompression,
    compression::{BincodeCompression, BytesCompression, Compressed, Compression, MaybeCompressed},
};

use building_blocks_core::prelude::*;

/// A `Compression` used for compressing `Chunk`s. It just uses the internal `FastArrayCompression` and clones the metadata.
#[derive(Copy, Clone)]
pub struct FastChunkCompression<N, T, Meta, B> {
    pub array_compression: FastArrayCompression<N, T, B>,
    marker: std::marker::PhantomData<Meta>,
}

impl<N, T, Meta, B> FastChunkCompression<N, T, Meta, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            array_compression: FastArrayCompression::new(bytes_compression),
            marker: Default::default(),
        }
    }
}

/// The target of `FastChunkCompression`. You probably want to use `Compressed<FastChunkCompression>` instead.
#[derive(Clone)]
pub struct FastCompressedChunk<N, T, Meta, B>
where
    T: Copy,
    B: BytesCompression,
    PointN<N>: IntegerPoint<N>,
{
    pub metadata: Meta, // metadata doesn't get compressed, hope it's small!
    pub compressed_array: Compressed<FastArrayCompression<N, T, B>>,
}

impl<N, T, Meta, B> Compression for FastChunkCompression<N, T, Meta, B>
where
    T: Copy,
    Meta: Clone,
    B: BytesCompression,
    PointN<N>: IntegerPoint<N>,
{
    type Data = Chunk<N, T, Meta>;
    type CompressedData = FastCompressedChunk<N, T, Meta, B>;

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

pub type BincodeChunkCompression<N, T, Meta, B> = BincodeCompression<Chunk<N, T, Meta>, B>;
pub type BincodeCompressedChunk<N, T, Meta, B> = Compressed<BincodeCompression<Chunk<N, T, Meta>, B>>;

pub type MaybeCompressedChunk<N, T, Meta, B> =
    MaybeCompressed<Chunk<N, T, Meta>, Compressed<FastChunkCompression<N, T, Meta, B>>>;

pub type MaybeCompressedChunkRef<'a, N, T, Meta, B> =
    MaybeCompressed<&'a Chunk<N, T, Meta>, &'a Compressed<FastChunkCompression<N, T, Meta, B>>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use super::*;
        use crate::$backend;

        pub type MaybeCompressedChunk2<T, Meta = (), B = $backend> =
            MaybeCompressedChunk<[i32; 2], T, Meta, B>;
        pub type MaybeCompressedChunk3<T, Meta = (), B = $backend> =
            MaybeCompressedChunk<[i32; 3], T, Meta, B>;
        pub type MaybeCompressedChunkRef2<'a, T, Meta = (), B = $backend> =
            MaybeCompressedChunkRef<'a, [i32; 2], T, Meta, B>;
        pub type MaybeCompressedChunkRef3<'a, T, Meta = (), B = $backend> =
            MaybeCompressedChunkRef<'a, [i32; 3], T, Meta, B>;
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
