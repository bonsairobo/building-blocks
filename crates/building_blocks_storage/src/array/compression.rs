use super::ArrayN;

use crate::compression::{BytesCompression, Compressed, Compression};

use building_blocks_core::prelude::*;

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
    T: Copy,
    PointN<N>: IntegerPoint<N>,
{
    type Data = ArrayN<N, T>;
    type CompressedData = FastCompressedArray<N, T, B>;

    // Compress the map in-memory using some `B: BytesCompression`.
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
