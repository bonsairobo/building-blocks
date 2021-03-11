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

/// An algorithm for:
///     1. compressing a specific type `Data` into type `Compressed`
///     2. decompressing `Compressed` back into `Data`
pub trait Compression: Sized {
    type Data;
    type CompressedData;

    fn compress(&self, data: &Self::Data) -> Compressed<Self>;
    fn decompress(compressed: &Self::CompressedData) -> Self::Data;
}

/// A value compressed with compression algorithm `A`.
#[derive(Clone, Deserialize, Serialize)]
pub struct Compressed<A>
where
    A: Compression,
{
    pub compressed_data: A::CompressedData,
    marker: std::marker::PhantomData<A>,
}

impl<T, A> Compressed<A>
where
    A: Compression<CompressedData = T>,
{
    pub fn new(compressed_data: A::CompressedData) -> Self {
        Self {
            compressed_data,
            marker: Default::default(),
        }
    }

    pub fn decompress(&self) -> A::Data {
        A::decompress(&self.compressed_data)
    }

    pub fn take(self) -> A::CompressedData {
        self.compressed_data
    }
}

/// A compression algorithm that acts directly on a slice of bytes.
pub trait BytesCompression {
    fn compress_bytes(&self, bytes: &[u8], compressed_bytes: impl std::io::Write);
    fn decompress_bytes(compressed_bytes: &[u8], bytes: &mut impl std::io::Write);
}

/// A value that is either compressed or decompressed.
pub enum MaybeCompressed<D, C> {
    Decompressed(D),
    Compressed(C),
}

impl<A: Compression> MaybeCompressed<A::Data, Compressed<A>> {
    pub fn as_decompressed(self) -> A::Data {
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

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Array3x1, IntoRawBytes};

    use building_blocks_core::prelude::*;

    #[cfg(feature = "lz4")]
    use crate::Lz4;
    #[cfg(feature = "snap")]
    use crate::Snappy;

    #[cfg(feature = "snap")]
    #[test]
    fn sphere_array_compression_rate_snappy() {
        sphere_array_compression_rate(Snappy, 32);
        sphere_array_compression_rate(Snappy, 64);
        sphere_array_compression_rate(Snappy, 128);
    }

    #[cfg(feature = "snap")]
    #[test]
    fn homogeneous_array_compression_rate_snappy() {
        homogeneous_array_compression_rate(Snappy, 32);
        homogeneous_array_compression_rate(Snappy, 64);
        homogeneous_array_compression_rate(Snappy, 128);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn sphere_array_compression_rate_lz4() {
        sphere_array_compression_rate(Lz4 { level: 10 }, 32);
        sphere_array_compression_rate(Lz4 { level: 10 }, 64);
        sphere_array_compression_rate(Lz4 { level: 10 }, 128);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn homogeneous_array_compression_rate_lz4() {
        homogeneous_array_compression_rate(Lz4 { level: 10 }, 32);
        homogeneous_array_compression_rate(Lz4 { level: 10 }, 64);
        homogeneous_array_compression_rate(Lz4 { level: 10 }, 128);
    }

    fn homogeneous_array_compression_rate<B: BytesCompression>(compression: B, side_length: i32) {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(side_length));
        let array = Array3x1::fill_with(extent, |_p| 0u16);
        array_compression_rate(&array, compression);
    }

    fn sphere_array_compression_rate<B: BytesCompression>(compression: B, side_length: i32) {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(side_length));
        let array = Array3x1::fill_with(extent, |p| if p.norm() > 50.0 { 0u16 } else { 1u16 });
        array_compression_rate(&array, compression);
    }

    fn array_compression_rate<B: BytesCompression>(array: &Array3x1<u16>, compression: B) {
        let source_size_bytes = array.extent().num_points() * 2;

        let mut compressed_bytes = Vec::new();
        compression.compress_bytes(array.into_raw_bytes(), &mut compressed_bytes);

        let compressed_size_bytes = compressed_bytes.len();

        test_print(&format!(
            "source = {} bytes, compressed = {} bytes; rate = {:.1}%\n",
            source_size_bytes,
            compressed_size_bytes,
            100.0 * (compressed_size_bytes as f32 / source_size_bytes as f32)
        ));
    }

    fn test_print(message: &str) {
        use std::io::Write;

        std::io::stdout()
            .lock()
            .write_all(message.as_bytes())
            .unwrap();
    }
}
