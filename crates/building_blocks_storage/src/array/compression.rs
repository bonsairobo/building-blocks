use crate::dev_prelude::{Array, Compression, FromBytesCompression};

use building_blocks_core::prelude::*;

use bytemuck::{bytes_of, bytes_of_mut};
use std::io;

/// A compression algorithm for arrays that avoid the overhead of serialization but ignores endianness and therefore isn't
/// portable.
#[derive(Clone, Copy, Debug)]
pub struct FastArrayCompression<N, C> {
    channels_compression: C,
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

impl<N, C> Compression for FastArrayCompression<N, C>
where
    PointN<N>: IntegerPoint<N>,
    C: Compression,
{
    type Data = Array<N, C::Data>;

    fn compress_to_writer(
        &self,
        data: &Self::Data,
        mut compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        // First write the extent.
        compressed_bytes.write_all(bytes_of(data.extent()))?;

        // Compress the channels.
        self.channels_compression
            .compress_to_writer(data.channels(), compressed_bytes)
    }

    fn decompress_from_reader(mut compressed_bytes: impl io::Read) -> io::Result<Self::Data> {
        // First read the extent.
        let mut extent = ExtentN::from_min_and_shape(PointN::ZERO, PointN::ZERO);
        compressed_bytes.read_exact(bytes_of_mut(&mut extent))?;

        // Decompress the channels.
        let channels = C::decompress_from_reader(compressed_bytes)?;

        Ok(Array::new(extent, channels))
    }
}

pub mod multichannel_aliases {
    use super::*;
    use crate::array::channels::multichannel::multichannel_aliases::*;

    pub type FastArrayCompressionNx1<N, By, A> =
        FastArrayCompression<N, FastChannelsCompression1<By, A>>;
    pub type FastArrayCompressionNx2<N, By, A, B> =
        FastArrayCompression<N, FastChannelsCompression2<By, A, B>>;
    pub type FastArrayCompressionNx3<N, By, A, B, C> =
        FastArrayCompression<N, FastChannelsCompression3<By, A, B, C>>;
    pub type FastArrayCompressionNx4<N, By, A, B, C, D> =
        FastArrayCompression<N, FastChannelsCompression4<By, A, B, C, D>>;
    pub type FastArrayCompressionNx5<N, By, A, B, C, D, E> =
        FastArrayCompression<N, FastChannelsCompression5<By, A, B, C, D, E>>;
    pub type FastArrayCompressionNx6<N, By, A, B, C, D, E, F> =
        FastArrayCompression<N, FastChannelsCompression6<By, A, B, C, D, E, F>>;
}

pub use multichannel_aliases::*;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::{Array3x1, BytesCompression};

    use crate::test_utilities::sphere_bit_array;
    use utilities::test::test_print;

    #[cfg(feature = "lz4")]
    use crate::compression::Lz4;
    #[cfg(feature = "snap")]
    use crate::compression::Snappy;

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
        let array = sphere_bit_array(side_length, 1u16, 0u16).0;
        array_compression_rate(&array, compression);
    }

    fn array_compression_rate<B: BytesCompression>(array: &Array3x1<u16>, bytes_compression: B) {
        let source_size_bytes = array.extent().num_points() * 2;

        let compression = FastArrayCompressionNx1::from_bytes_compression(bytes_compression);

        let compressed_bytes = compression.compress(array).take_bytes();

        let compressed_size_bytes = compressed_bytes.len();

        test_print(&format!(
            "source = {} bytes, compressed = {} bytes; rate = {:.1}%\n",
            source_size_bytes,
            compressed_size_bytes,
            100.0 * (compressed_size_bytes as f32 / source_size_bytes as f32)
        ));
    }
}
