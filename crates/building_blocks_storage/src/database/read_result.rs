use crate::dev_prelude::{ChunkKey, Compression, DatabaseKey};

use futures::future::join_all;
use sled::IVec;

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

    pub fn append(&mut self, mut result: Self) {
        self.key_value_pairs.append(&mut result.key_value_pairs)
    }

    pub fn take_raw(self) -> Vec<(IVec, IVec)> {
        self.key_value_pairs
    }

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
