use super::{BytesCompression, Compression};

use serde::{de::DeserializeOwned, Serialize};
use std::io;

/// Run some compression algorithm `A` after bincode serializing a type `T`. This provides a decent
/// default compression for any serializable type.
pub struct BincodeCompression<T, A> {
    pub compression: A,
    marker: std::marker::PhantomData<T>,
}

impl<T, A> Clone for BincodeCompression<T, A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        Self {
            compression: self.compression.clone(),
            marker: Default::default(),
        }
    }
}

impl<T, A> Copy for BincodeCompression<T, A> where A: Copy {}

impl<T, A> BincodeCompression<T, A> {
    pub fn new(compression: A) -> Self {
        Self {
            compression,
            marker: Default::default(),
        }
    }
}

impl<T, A> Compression for BincodeCompression<T, A>
where
    T: DeserializeOwned + Serialize,
    A: BytesCompression,
{
    type Data = T;

    fn compress_to_writer(
        &self,
        data: &Self::Data,
        compressed_bytes: impl io::Write,
    ) -> io::Result<()> {
        self.compression.compress_bytes(
            bincode::serialize(data).unwrap().as_slice(),
            compressed_bytes,
        )
    }

    fn decompress_from_reader(compressed_bytes: impl io::Read) -> io::Result<Self::Data> {
        let mut decompressed_bytes = Vec::new();
        A::decompress_bytes(compressed_bytes, &mut decompressed_bytes)?;

        Ok(bincode::deserialize(&decompressed_bytes).unwrap())
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(all(test, feature = "snap"))]
mod tests {
    use super::*;
    use crate::prelude::Snappy;
    use serde::Deserialize;

    #[derive(Clone, Debug, Eq, Deserialize, Serialize, PartialEq)]
    struct Foo(Vec<u8>);

    #[test]
    fn compress_and_decompress_serializable_type() {
        let foo = Foo((0u8..100).collect());

        let compression = BincodeCompression::new(Snappy);
        let compressed = compression.compress(&foo);
        let decompressed_foo = compressed.decompress();

        assert_eq!(foo, decompressed_foo);
    }
}
