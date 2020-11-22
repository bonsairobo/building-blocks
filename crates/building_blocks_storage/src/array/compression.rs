use super::ArrayN;

use building_blocks_core::prelude::*;

use compressible_map::{BytesCompression, Compressed, Compression};

/// A compression algorithm for arrays that avoid the overhead of serialization but ignores
/// endianness and therefore isn't portable.
#[derive(Clone, Copy, Debug)]
pub struct FastArrayCompression<N, T, B> {
    pub bytes_compression: B,
    marker: std::marker::PhantomData<(N, T)>,
}

impl<N, T, B> FastArrayCompression<N, T, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            bytes_compression,
            marker: Default::default(),
        }
    }
}

/// A compressed `ArrayN` that decompresses quickly but only on the same platform where it was
/// compressed.
#[derive(Clone)]
pub struct FastCompressedArray<N, T, B> {
    compressed_bytes: Vec<u8>,
    extent: ExtentN<N>,
    marker: std::marker::PhantomData<(T, B)>,
}

impl<N, T, B> FastCompressedArray<N, T, B> {
    pub fn compressed_bytes(&self) -> &[u8] {
        &self.compressed_bytes
    }

    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }
}

impl<N, T, B> Compression for FastArrayCompression<N, T, B>
where
    B: BytesCompression,
    T: Copy, // Copy is important so we don't serialize a vector of non-POD type
    ExtentN<N>: IntegerExtent<N>,
{
    type Data = ArrayN<N, T>;
    type CompressedData = FastCompressedArray<N, T, B>;

    // Compress the map in-memory using some `A: BytesCompression`.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without
    // accounting for endianness. This is not compatible across platforms.
    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        let mut compressed_bytes = Vec::new();
        self.bytes_compression
            .compress_bytes(data.bytes_slice(), &mut compressed_bytes);

        Compressed::new(FastCompressedArray {
            extent: data.extent,
            compressed_bytes,
            marker: Default::default(),
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        let num_points = compressed.extent.num_points();

        // Allocate the vector with element type T so the alignment is correct.
        let mut decompressed_values: Vec<T> = Vec::with_capacity(num_points);
        unsafe { decompressed_values.set_len(num_points) };
        let mut decompressed_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                decompressed_values.as_mut_ptr() as *mut u8,
                num_points * core::mem::size_of::<T>(),
            )
        };
        B::decompress_bytes(&compressed.compressed_bytes, &mut decompressed_bytes);

        ArrayN::new(compressed.extent, decompressed_values)
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
    use crate::{Array3, Lz4};

    #[cfg(feature = "snappy")]
    use crate::Snappy;

    #[cfg(feature = "snappy")]
    #[test]
    fn sphere_array_compression_rate_snappy() {
        sphere_array_compression_rate(Snappy);
    }

    #[test]
    fn sphere_array_compression_rate_lz4() {
        sphere_array_compression_rate(Lz4 { level: 10 });
    }

    fn sphere_array_compression_rate<B: BytesCompression>(compression: B) {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([64; 3]));
        let array = Array3::fill_with(extent, |p| if p.norm() > 50.0 { 0u16 } else { 1u16 });

        let source_size_bytes = array.extent().num_points() * 2;

        let compression = FastArrayCompression::new(compression);
        let compressed_array = compression.compress(&array);

        let compressed_size_bytes = compressed_array.compressed_data.compressed_bytes().len();

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
