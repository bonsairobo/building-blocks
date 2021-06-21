use super::BytesCompression;

use serde::{Deserialize, Serialize};
use std::io;

/// The [Snappy compression algorithm](https://en.wikipedia.org/wiki/Snappy_(compression)).
/// Uses a pure Rust implementation, making it suitable for use with the WASM target.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Snappy;

impl BytesCompression for Snappy {
    fn compress_bytes(
        &self,
        mut bytes: impl io::Read,
        compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        let mut encoder = snap::write::FrameEncoder::new(compressed_bytes);
        io::copy(&mut bytes, &mut encoder)?;
        encoder.into_inner().expect("failed to flush the writer");
        Ok(())
    }

    fn decompress_bytes(
        compressed_bytes: impl io::Read,
        mut bytes: impl io::Write,
    ) -> io::Result<()> {
        let mut decoder = snap::read::FrameDecoder::new(compressed_bytes);
        io::copy(&mut decoder, &mut bytes)?;
        Ok(())
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_and_decompress_serializable_type() {
        let bytes: Vec<u8> = (0u8..100).collect();

        let mut compressed_bytes = Vec::new();
        Snappy
            .compress_bytes(bytes.as_slice(), &mut compressed_bytes)
            .unwrap();
        let mut decompressed_bytes = Vec::new();
        Snappy::decompress_bytes(compressed_bytes.as_slice(), &mut decompressed_bytes).unwrap();

        assert_eq!(bytes, decompressed_bytes);
    }
}
