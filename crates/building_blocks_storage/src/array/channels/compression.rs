use crate::{BytesCompression, Channel, Compressed, Compression, FromBytesCompression};

use bytemuck::{cast_slice, cast_slice_mut, Pod};

/// Compresses a tuple of `Channel`s into a tuple of `FastCompressedChannel`s.
pub struct FastChannelsCompression<By, Chan> {
    bytes_compression: By,
    marker: std::marker::PhantomData<Chan>,
}

impl<By, Chan> Clone for FastChannelsCompression<By, Chan>
where
    By: Clone,
{
    fn clone(&self) -> Self {
        Self {
            bytes_compression: self.bytes_compression.clone(),
            marker: Default::default(),
        }
    }
}

impl<By, Chan> Copy for FastChannelsCompression<By, Chan> where By: Copy {}

impl<By, Chan> FastChannelsCompression<By, Chan> {
    pub fn new(bytes_compression: By) -> Self {
        Self {
            bytes_compression,
            marker: Default::default(),
        }
    }

    pub fn bytes_compression(&self) -> &By {
        &self.bytes_compression
    }
}

impl<By, Chan> FromBytesCompression<By> for FastChannelsCompression<By, Chan> {
    fn from_bytes_compression(bytes_compression: By) -> Self {
        Self::new(bytes_compression)
    }
}

/// A compressed `Channel` that decompresses quickly but only on the same platform where it was compressed.
#[derive(Clone)]
pub struct FastCompressedChannel<T> {
    compressed_bytes: Vec<u8>,
    decompressed_length: usize, // TODO: we should be able to remove this with some refactoring of the Compression trait
    marker: std::marker::PhantomData<T>,
}

impl<T> FastCompressedChannel<T> {
    pub fn compressed_bytes(&self) -> &[u8] {
        &self.compressed_bytes
    }
}

impl<By, T> Compression for FastChannelsCompression<By, Channel<T>>
where
    By: BytesCompression,
    T: Pod,
{
    type Data = Channel<T>;
    type CompressedData = FastCompressedChannel<T>;

    // Compress the map using some `B: BytesCompression`.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without accounting for endianness. This is
    // not compatible across platforms.
    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        let mut compressed_bytes = Vec::new();
        self.bytes_compression
            .compress_bytes(cast_slice(data.store().as_slice()), &mut compressed_bytes);

        Compressed::new(FastCompressedChannel {
            compressed_bytes,
            decompressed_length: data.store().len(),
            marker: Default::default(),
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        let num_values = compressed.decompressed_length;

        // Allocate the vector with element type T so the alignment is correct.
        let mut decompressed_values: Vec<T> = Vec::with_capacity(num_values);
        unsafe { decompressed_values.set_len(num_values) };
        By::decompress_bytes(
            &compressed.compressed_bytes,
            &mut cast_slice_mut(decompressed_values.as_mut_slice()),
        );

        Channel::new(decompressed_values)
    }
}
