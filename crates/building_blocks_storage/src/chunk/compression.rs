use crate::{chunk::ChunkNode, dev_prelude::Compression};

use bytemuck::bytes_of;
use std::io;

/// A compression algorithm for `ChunkNode`s, the container of user chunk data.
#[derive(Clone, Copy, Debug)]
pub struct FastChunkCompression<C> {
    user_chunk_compression: C,
}

impl<C> FastChunkCompression<C> {
    pub fn new(user_chunk_compression: C) -> Self {
        Self {
            user_chunk_compression,
        }
    }

    pub fn user_chunk_compression(&self) -> &C {
        &self.user_chunk_compression
    }
}

impl<C> Compression for FastChunkCompression<C>
where
    C: Compression,
{
    type Data = ChunkNode<C::Data>;

    fn compress_to_writer(
        &self,
        data: &Self::Data,
        mut compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        // First write the child_bitmask.
        compressed_bytes.write_all(bytes_of(&data.child_mask()))?;
        // Compress the user data if it exists, otherwise write a single 0 byte indicating no data left.
        if let Some(user_chunk) = data.user_chunk() {
            compressed_bytes.write_all(&[1])?; // There is data.
            self.user_chunk_compression
                .compress_to_writer(user_chunk, compressed_bytes)?;
        } else {
            compressed_bytes.write_all(&[0])?; // There is no data.
        }
        Ok(())
    }

    fn decompress_from_reader(mut compressed_bytes: impl io::Read) -> io::Result<Self::Data> {
        // First two bytes are the child bitmask and a "has data" flag.
        let mut two_bytes = [0u8; 2];
        compressed_bytes.read_exact(&mut two_bytes)?;
        let [child_mask, has_data] = two_bytes;
        // Check if there is more data.
        let user_chunk = if has_data == 1 {
            Some(C::decompress_from_reader(compressed_bytes)?)
        } else {
            None
        };
        Ok(ChunkNode::new(user_chunk, child_mask))
    }
}
