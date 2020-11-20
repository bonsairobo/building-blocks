use super::ArrayN;

use building_blocks_core::prelude::*;

use compressible_map::{Compressible, Decompressible};

//  ██████╗ ██████╗ ███╗   ███╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗
// ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║
// ██║     ██║   ██║██╔████╔██║██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║
// ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║
// ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║
//  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

/// A compression algorithm that decompresses faster than similar algorithms (LZO, LZF, QuickLZ,
/// etc.) but is primarily optimized for 64-bit x86-compatible processors.
///
/// Unlike the LZ4 backend, `FastSnappy` uses a compression algorithm implemented entirely in
/// rust, which means it can be compiled for targets like `wasm32-unknown-unknown`.
#[derive(Clone, Copy, Debug)]
pub struct FastSnappy;

/// A compressed `ArrayN` that decompresses faster than similar algorithms (LZO, LZF, QuickLZ,
/// etc.) but is primarily optimized for 64-bit x86-compatible processors.
#[derive(Clone)]
pub struct FastSnappyCompressedArrayN<N, T> {
    pub compressed_bytes: Vec<u8>,
    pub extent: ExtentN<N>,
    marker: std::marker::PhantomData<T>,
}

impl<N, T> FastSnappyCompressedArrayN<N, T> {
    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }
}

impl<N, T> Decompressible<FastSnappy> for FastSnappyCompressedArrayN<N, T>
where
    T: Copy, // Copy is important so we don't serialize a vector of non-POD type
    ExtentN<N>: IntegerExtent<N>,
{
    type Decompressed = ArrayN<N, T>;

    fn decompress(&self) -> Self::Decompressed {
        let num_points = self.extent.num_points();

        let mut decoder = snap::read::FrameDecoder::new(self.compressed_bytes.as_slice());
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

impl<N, T> Compressible<FastSnappy> for ArrayN<N, T>
where
    T: Copy, // Copy is important so we don't serialize a vector of non-POD type
    ExtentN<N>: IntegerExtent<N>,
{
    type Compressed = FastSnappyCompressedArrayN<N, T>;

    // Compress the map in-memory using the snappy algorithm.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without
    // accounting for endianness. This is not compatible across platforms.
    fn compress(&self, _params: FastSnappy) -> FastSnappyCompressedArrayN<N, T> {
        let values_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.values.as_ptr() as *const u8,
                self.values.len() * core::mem::size_of::<T>(),
            )
        };
        let mut encoder = snap::write::FrameEncoder::new(Vec::new());

        std::io::copy(&mut std::io::Cursor::new(values_slice), &mut encoder).unwrap();
        let compressed_bytes = encoder.into_inner().expect("failed to get the underlying stream");

        FastSnappyCompressedArrayN {
            extent: self.extent,
            compressed_bytes,
            marker: Default::default(),
        }
    }
}
