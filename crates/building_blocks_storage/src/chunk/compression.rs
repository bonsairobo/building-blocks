use crate::{
    ArrayN, BincodeCompression, BytesCompression, Compressed, Compression, FastArrayCompression,
    MaybeCompressed,
};

use building_blocks_core::prelude::*;

/// A `Compression` used for compressing `Chunk`s. It just uses the internal `FastArrayCompression` and clones the metadata.
#[derive(Copy, Clone)]
pub struct FastChunkCompression<N, T, B> {
    pub array_compression: FastArrayCompression<N, T, B>,
}

impl<N, T, B> FastChunkCompression<N, T, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            array_compression: FastArrayCompression::new(bytes_compression),
        }
    }
}

/// The target of `FastChunkCompression`. You probably want to use `Compressed<FastChunkCompression>` instead.
#[derive(Clone)]
pub struct FastCompressedChunk<N, T, B>
where
    T: 'static + Copy,
    B: BytesCompression,
    PointN<N>: IntegerPoint<N>,
{
    pub compressed_array: Compressed<FastArrayCompression<N, T, B>>,
}

impl<N, T, B> Compression for FastChunkCompression<N, T, B>
where
    T: 'static + Copy,
    B: BytesCompression,
    PointN<N>: IntegerPoint<N>,
{
    type Data = ArrayN<N, T>;
    type CompressedData = FastCompressedChunk<N, T, B>;

    // PERF: cloning the metadata is unfortunate

    fn compress(&self, chunk: &Self::Data) -> Compressed<Self> {
        Compressed::new(FastCompressedChunk {
            compressed_array: self.array_compression.compress(chunk),
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        compressed.compressed_array.decompress()
    }
}

pub type BincodeChunkCompression<Ch, B> = BincodeCompression<Ch, B>;
pub type BincodeCompressedChunk<Ch, B> = Compressed<BincodeCompression<Ch, B>>;

pub type MaybeCompressedChunkNx1<N, T, B> =
    MaybeCompressed<ArrayN<N, T>, Compressed<FastChunkCompression<N, T, B>>>;

pub type MaybeCompressedChunkRefNx1<'a, N, T, B> =
    MaybeCompressed<&'a ArrayN<N, T>, &'a Compressed<FastChunkCompression<N, T, B>>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::$backend;

        pub type MaybeCompressedChunk2x1<T, B = $backend> = MaybeCompressedChunkNx1<[i32; 2], T, B>;
        pub type MaybeCompressedChunk3x1<T, B = $backend> = MaybeCompressedChunkNx1<[i32; 3], T, B>;
        pub type MaybeCompressedChunkRef2x1<'a, T, B = $backend> =
            MaybeCompressedChunkRefNx1<'a, [i32; 2], T, B>;
        pub type MaybeCompressedChunkRef3x1<'a, T, B = $backend> =
            MaybeCompressedChunkRefNx1<'a, [i32; 3], T, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if we only use one, then we want to have these aliases refer to the choice we
// made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);
