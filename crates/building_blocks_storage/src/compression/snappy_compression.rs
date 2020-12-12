use super::BytesCompression;

use serde::{Deserialize, Serialize};

/// The [Snappy compression algorithm](https://en.wikipedia.org/wiki/Snappy_(compression)).
/// Uses a pure Rust implementation, making it suitable for use with the WASM target.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Snappy;

impl BytesCompression for Snappy {
    fn compress_bytes(&self, bytes: &[u8], compressed_bytes: impl std::io::Write) {
        let mut encoder = snap::write::FrameEncoder::new(compressed_bytes);
        std::io::copy(&mut std::io::Cursor::new(bytes), &mut encoder).unwrap();
        encoder.into_inner().expect("failed to flush the writer");
    }

    fn decompress_bytes(compressed_bytes: &[u8], bytes: &mut impl std::io::Write) {
        let mut decoder = snap::read::FrameDecoder::new(compressed_bytes);
        std::io::copy(&mut decoder, bytes).unwrap();
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_and_decompress_serializable_type() {
        let bytes: Vec<u8> = (0u8..100).collect();

        let mut compressed_bytes = Vec::new();
        Snappy.compress_bytes(&bytes, &mut compressed_bytes);
        let mut decompressed_bytes = Vec::new();
        Snappy::decompress_bytes(&compressed_bytes, &mut decompressed_bytes);

        assert_eq!(bytes, decompressed_bytes);
    }
}
