mod compressed_bincode;

#[cfg(feature = "lz4")]
mod lz4_compression;
#[cfg(feature = "snap")]
mod snappy_compression;

pub use compressed_bincode::BincodeCompression;

#[cfg(feature = "lz4")]
pub use lz4_compression::Lz4;
#[cfg(feature = "snap")]
pub use snappy_compression::Snappy;

use serde::{Deserialize, Serialize};
use std::io;

/// An algorithm for:
///     1. compressing a specific type `Data` into raw bytes
///     2. decompressing raw bytes back into `Data`
pub trait Compression: Sized {
    type Data;

    fn compress_to_writer(
        &self,
        data: &Self::Data,
        compressed_bytes: impl io::Write,
    ) -> io::Result<()>;

    fn decompress_from_reader(compressed_bytes: impl io::Read) -> io::Result<Self::Data>;

    /// To preserve type information. prefer this method over `compress_to_writer`.
    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        Compressed::new(self, data)
    }
}

pub trait FromBytesCompression<B> {
    fn from_bytes_compression(bytes_compression: B) -> Self;
}

/// A wrapper for bytes from compression algorithm `A`. This is slightly safer than manually calling `decompress` on any byte
/// slice, since it remembers the original data type.
#[derive(Clone, Deserialize, Serialize)]
pub struct Compressed<A> {
    pub compressed_bytes: Vec<u8>,
    marker: std::marker::PhantomData<A>,
}

impl<A> Compressed<A>
where
    A: Compression,
{
    pub fn new(compression: &A, data: &A::Data) -> Self {
        let mut compressed_bytes = Vec::new();
        A::compress_to_writer(compression, data, &mut compressed_bytes).unwrap();

        Self {
            compressed_bytes,
            marker: Default::default(),
        }
    }

    pub fn decompress(&self) -> A::Data {
        A::decompress_from_reader(self.compressed_bytes.as_slice()).unwrap()
    }

    pub fn take_bytes(self) -> Vec<u8> {
        self.compressed_bytes
    }
}

/// A compression algorithm that reads a stream of bytes.
pub trait BytesCompression {
    fn compress_bytes(
        &self,
        bytes: impl io::Read,
        compressed_bytes: impl io::Write,
    ) -> io::Result<()>;

    fn decompress_bytes(compressed_bytes: impl io::Read, bytes: impl io::Write) -> io::Result<()>;
}

/// A value that is either compressed or decompressed.
pub enum MaybeCompressed<D, C> {
    Decompressed(D),
    Compressed(C),
}

impl<A: Compression> MaybeCompressed<A::Data, Compressed<A>> {
    pub fn into_decompressed(self) -> A::Data {
        match self {
            MaybeCompressed::Compressed(c) => c.decompress(),
            MaybeCompressed::Decompressed(d) => d,
        }
    }

    pub fn unwrap_decompressed(self) -> A::Data {
        match self {
            MaybeCompressed::Compressed(_) => panic!("Must be decompressed"),
            MaybeCompressed::Decompressed(d) => d,
        }
    }
}
