use super::BytesCompression;

use serde::{Deserialize, Serialize};

/// The [LZ4 compression algorithm](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)).
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Lz4 {
    /// The compression level, from 0 to 10. 0 is fastest and least aggressive. 10 is slowest and
    /// most aggressive.
    pub level: u32,
}

impl BytesCompression for Lz4 {
    fn compress_bytes(&self, bytes: &[u8], compressed_bytes: impl std::io::Write) {
        let mut encoder = lz4::EncoderBuilder::new()
            .level(self.level)
            .build(compressed_bytes)
            .unwrap();
        std::io::copy(&mut std::io::Cursor::new(bytes), &mut encoder).unwrap();
        let (_output, _result) = encoder.finish();
    }

    fn decompress_bytes(compressed_bytes: &[u8], bytes: &mut impl std::io::Write) {
        let mut decoder = lz4::Decoder::new(compressed_bytes).unwrap();
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
        Lz4 { level: 10 }.compress_bytes(&bytes, &mut compressed_bytes);
        let mut decompressed_bytes = Vec::new();
        Lz4::decompress_bytes(&compressed_bytes, &mut decompressed_bytes);

        assert_eq!(bytes, decompressed_bytes);
    }
}
