use super::ArrayN;

use building_blocks_core::prelude::*;

use compressible_map::{Compressible, Decompressible};

/// A compression algorithm that decompresses quickly, but only on the same platform where it was
/// compressed.
#[derive(Clone, Copy, Debug)]
pub struct FastLz4 {
    pub level: u32,
}

/// A compressed `ArrayN` that decompresses quickly, but only on the same platform where it
/// was compressed.
#[derive(Clone)]
pub struct FastLz4CompressedArrayN<N, T> {
    pub compressed_bytes: Vec<u8>,
    pub extent: ExtentN<N>,
    marker: std::marker::PhantomData<T>,
}

impl<N, T> FastLz4CompressedArrayN<N, T> {
    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }
}

impl<N, T> Decompressible<FastLz4> for FastLz4CompressedArrayN<N, T>
where
    T: Copy, // Copy is important so we don't serialize a vector of non-POD type
    ExtentN<N>: IntegerExtent<N>,
{
    type Decompressed = ArrayN<N, T>;

    fn decompress(&self) -> Self::Decompressed {
        let num_points = self.extent.num_points();

        let mut decoder = lz4::Decoder::new(self.compressed_bytes.as_slice()).unwrap();
        // Allocate the vector with element type T so the alignment is correct.
        let mut decompressed_values: Vec<T> = Vec::with_capacity(num_points);
        unsafe { decompressed_values.set_len(num_points) };
        let mut decompressed_slice = unsafe {
            std::slice::from_raw_parts_mut(
                decompressed_values.as_mut_ptr() as *mut u8,
                num_points * core::mem::size_of::<T>(),
            )
        };
        std::io::copy(&mut decoder, &mut decompressed_slice).unwrap();

        ArrayN::new(self.extent, decompressed_values)
    }
}

impl<N, T> Compressible<FastLz4> for ArrayN<N, T>
where
    T: Copy, // Copy is important so we don't serialize a vector of non-POD type
    ExtentN<N>: IntegerExtent<N>,
{
    type Compressed = FastLz4CompressedArrayN<N, T>;

    // Compress the map in-memory using the LZ4 algorithm.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without
    // accounting for endianness. This is not compatible across platforms.
    fn compress(&self, params: FastLz4) -> FastLz4CompressedArrayN<N, T> {
        let mut compressed_bytes = Vec::new();
        let values_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.values.as_ptr() as *const u8,
                self.values.len() * core::mem::size_of::<T>(),
            )
        };
        let mut encoder = lz4::EncoderBuilder::new()
            .level(params.level)
            .build(&mut compressed_bytes)
            .unwrap();

        std::io::copy(&mut std::io::Cursor::new(values_slice), &mut encoder).unwrap();
        let (_output, _result) = encoder.finish();

        FastLz4CompressedArrayN {
            extent: self.extent,
            compressed_bytes,
            marker: Default::default(),
        }
    }
}
