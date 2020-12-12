use super::{BytesCompression, Compressed, Compression};

use serde::{de::DeserializeOwned, Serialize};

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
    type CompressedData = Vec<u8>;

    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        let mut compressed_bytes = Vec::new();
        self.compression
            .compress_bytes(&bincode::serialize(data).unwrap(), &mut compressed_bytes);

        Compressed::new(compressed_bytes)
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        let mut decompressed_bytes = Vec::new();
        A::decompress_bytes(compressed, &mut decompressed_bytes);

        bincode::deserialize(&decompressed_bytes).unwrap()
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(all(test, feature = "snap"))]
mod tests {
    use super::*;
    use crate::Snappy;
    use serde::Deserialize;

    #[derive(Clone, Debug, Eq, Deserialize, Serialize, PartialEq)]
    struct Foo(Vec<u8>);

    #[test]
    fn compress_and_decompress_serializable_type() {
        let foo = Foo((0u8..100).collect());

        let compression = BincodeCompression::new(Snappy);
        let compressed_bytes = compression.compress(&foo);
        let decompressed_foo = compressed_bytes.decompress();

        assert_eq!(foo, decompressed_foo);
    }
}
