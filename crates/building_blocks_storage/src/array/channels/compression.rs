use crate::dev_prelude::{BytesCompression, Channel, Compression, FromBytesCompression};

use bytemuck::{bytes_of, bytes_of_mut, cast_slice, cast_slice_mut, Pod};
use std::io;

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

impl<By, T> Compression for FastChannelsCompression<By, Channel<T>>
where
    By: BytesCompression,
    T: Pod,
{
    type Data = Channel<T>;

    // Compress the map using some `B: BytesCompression`.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without accounting for endianness. This is
    // not compatible across platforms.
    fn compress_to_writer(
        &self,
        data: &Self::Data,
        mut compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        // Start with the number of values in the channel so we can allocate that up front during decompression.
        compressed_bytes.write_all(bytes_of(&data.store().len()))?;

        // Compress the values.
        self.bytes_compression
            .compress_bytes(cast_slice(data.store().as_slice()), compressed_bytes)
    }

    fn decompress_from_reader(mut compressed_bytes: impl io::Read) -> io::Result<Self::Data> {
        // Extract the number of values in the original channel.
        let mut num_values = 0usize;
        compressed_bytes.read_exact(bytes_of_mut(&mut num_values))?;

        // Allocate the vector with element type T so the alignment is correct.
        let mut decompressed_values: Vec<T> = Vec::with_capacity(num_values);
        unsafe { decompressed_values.set_len(num_values) };

        // Decompress the values by consuming the rest of the bytes.
        By::decompress_bytes(
            compressed_bytes,
            cast_slice_mut(decompressed_values.as_mut_slice()),
        )?;

        Ok(Channel::new(decompressed_values))
    }
}
