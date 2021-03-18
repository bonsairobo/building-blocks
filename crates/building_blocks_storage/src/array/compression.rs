use crate::{
    Array, ArrayNx1, BincodeCompression, Compressed, Compression, FastChannelsCompression1,
    FromBytesCompression,
};

use building_blocks_core::prelude::*;

/// A compression algorithm for arrays that avoid the overhead of serialization but ignores endianness and therefore isn't
/// portable.
#[derive(Clone, Copy, Debug)]
pub struct FastArrayCompression<N, C> {
    pub channels_compression: C,
    marker: std::marker::PhantomData<N>,
}

impl<N, C> FastArrayCompression<N, C> {
    pub fn new(channels_compression: C) -> Self {
        Self {
            channels_compression,
            marker: Default::default(),
        }
    }

    pub fn channels_compression(&self) -> &C {
        &self.channels_compression
    }
}

impl<N, C, B> FromBytesCompression<B> for FastArrayCompression<N, C>
where
    C: FromBytesCompression<B>,
{
    fn from_bytes_compression(bytes_compression: B) -> Self {
        Self::new(C::from_bytes_compression(bytes_compression))
    }
}

/// A compressed `Array` that decompresses quickly but only on the same platform where it was compressed.
#[derive(Clone)]
pub struct FastCompressedArray<N, C>
where
    C: Compression,
{
    compressed_channels: C::CompressedData,
    extent: ExtentN<N>,
}

impl<N, C> FastCompressedArray<N, C>
where
    C: Compression,
{
    pub fn compressed_channels(&self) -> &C::CompressedData {
        &self.compressed_channels
    }

    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }

    pub fn into_parts(self) -> (C::CompressedData, ExtentN<N>) {
        (self.compressed_channels, self.extent)
    }
}

impl<N, C> Compression for FastArrayCompression<N, C>
where
    PointN<N>: IntegerPoint<N>,
    C: Compression,
{
    type Data = Array<N, C::Data>;
    type CompressedData = FastCompressedArray<N, C>;

    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        let compressed_channels = self.channels_compression.compress(data.channels()).take();

        Compressed::new(FastCompressedArray {
            compressed_channels,
            extent: data.extent,
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        Array::new(
            compressed.extent,
            C::decompress(&compressed.compressed_channels),
        )
    }
}

pub type FastArrayCompressionNx1<N, T, B> = FastArrayCompression<N, FastChannelsCompression1<B, T>>;

pub type BincodeArrayCompression<N, T, B> = BincodeCompression<ArrayNx1<N, T>, B>;
pub type BincodeCompressedArray<N, T, B> = Compressed<BincodeArrayCompression<N, T, B>>;

macro_rules! define_conditional_aliases {
    ($backend:ident) => {
        use crate::{$backend, MaybeCompressed};

        pub type MaybeCompressedArrayNx1<N, T, B = $backend> =
            MaybeCompressed<ArrayNx1<N, T>, Compressed<FastArrayCompressionNx1<N, T, B>>>;
        pub type MaybeCompressedArray2x1<T, B = $backend> = MaybeCompressedArrayNx1<[i32; 2], T, B>;
        pub type MaybeCompressedArray3x1<T, B = $backend> = MaybeCompressedArrayNx1<[i32; 3], T, B>;

        pub type MaybeCompressedArrayRefN<'a, N, T, B = $backend> =
            MaybeCompressed<&'a ArrayNx1<N, T>, &'a Compressed<FastArrayCompressionNx1<N, T, B>>>;
        pub type MaybeCompressedArrayRef2<'a, T, B = $backend> =
            MaybeCompressedArrayRefN<'a, [i32; 2], T, B>;
        pub type MaybeCompressedArrayRef3<'a, T, B = $backend> =
            MaybeCompressedArrayRefN<'a, [i32; 3], T, B>;
    };
}

// LZ4 and Snappy are not mutually exclusive, but if we only use one, then we want to have these aliases refer to the choice we
// made.
#[cfg(all(feature = "lz4", not(feature = "snap")))]
define_conditional_aliases!(Lz4);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
define_conditional_aliases!(Snappy);

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Array3x1, BytesCompression};

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

    fn array_compression_rate<B: BytesCompression>(array: &Array3x1<u16>, bytes_compression: B) {
        let source_size_bytes = array.extent().num_points() * 2;

        let compression = FastArrayCompressionNx1::from_bytes_compression(bytes_compression);

        let compressed_array = compression.compress(array).take();

        let compressed_size_bytes = compressed_array
            .compressed_channels()
            .compressed_bytes()
            .len();

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
