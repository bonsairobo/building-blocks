use super::BytesCompression;

use std::io;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The [LZ4 compression algorithm](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lz4 {
    /// The compression level, from 0 to 10. 0 is fastest and least aggressive. 10 is slowest and
    /// most aggressive.
    pub level: u32,
}

impl BytesCompression for Lz4 {
    fn compress_bytes(
        &self,
        mut bytes: impl io::Read,
        compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        let mut encoder = lz4::EncoderBuilder::new()
            .level(self.level)
            .build(compressed_bytes)?;
        io::copy(&mut bytes, &mut encoder)?;
        let (_output, result) = encoder.finish();

        result
    }

    fn decompress_bytes(
        compressed_bytes: impl io::Read,
        mut bytes: impl io::Write,
    ) -> io::Result<()> {
        let mut decoder = lz4::Decoder::new(compressed_bytes)?;
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
        Lz4 { level: 10 }
            .compress_bytes(bytes.as_slice(), &mut compressed_bytes)
            .unwrap();
        let mut decompressed_bytes = Vec::new();
        Lz4::decompress_bytes(compressed_bytes.as_slice(), &mut decompressed_bytes).unwrap();

        assert_eq!(bytes, decompressed_bytes);
    }
}
