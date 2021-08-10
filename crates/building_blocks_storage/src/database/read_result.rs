use crate::dev_prelude::{ChunkKey, Compression, DatabaseKey};

use futures::future::join_all;
use sled::IVec;

/// A wrapper around key-value pairs read from a `ChunkDb`.
pub struct ReadResult<Compr> {
    pub(crate) key_value_pairs: Vec<(IVec, IVec)>,
    marker: std::marker::PhantomData<Compr>,
}

impl<Compr> Default for ReadResult<Compr> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<Compr> ReadResult<Compr> {
    pub(crate) fn new(key_value_pairs: Vec<(IVec, IVec)>) -> Self {
        Self {
            key_value_pairs,
            marker: Default::default(),
        }
    }

    pub(crate) fn append(&mut self, mut result: Self) {
        self.key_value_pairs.append(&mut result.key_value_pairs)
    }

    /// Take the key-value pairs where keys and values are left in a raw byte format.
    pub fn take_with_raw_key_values<N>(self) -> Vec<(IVec, IVec)> {
        self.key_value_pairs
    }

    /// Take the key-value pairs where values are left in a raw byte format.
    pub fn take_with_raw_values<N>(self) -> impl Iterator<Item = (ChunkKey<N>, IVec)>
    where
        ChunkKey<N>: DatabaseKey<N>,
    {
        self.key_value_pairs.into_iter().map(|(k, v)| {
            (
                ChunkKey::from_ord_key(ChunkKey::ord_key_from_be_bytes(&k)),
                v,
            )
        })
    }

    /// Concurrently decompress all values, calling `chunk_rx` on each key-value pair.
    pub async fn decompress<N, F>(self, mut chunk_rx: F)
    where
        ChunkKey<N>: DatabaseKey<N>,
        Compr: Compression,
        F: FnMut(ChunkKey<N>, Compr::Data),
    {
        for batch in self.key_value_pairs.chunks(16) {
            for (chunk_key, chunk) in
                join_all(batch.iter().map(|(key, compressed_chunk)| async move {
                    let ord_key = ChunkKey::<N>::ord_key_from_be_bytes(key.as_ref());
                    let chunk_key = ChunkKey::<N>::from_ord_key(ord_key);

                    let chunk = Compr::decompress_from_reader(compressed_chunk.as_ref()).unwrap();

                    (chunk_key, chunk)
                }))
                .await
            {
                chunk_rx(chunk_key, chunk);
            }
        }
    }
}
