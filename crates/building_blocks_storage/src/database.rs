mod chunk_db;
mod key;

#[cfg(feature = "sled-snapshots")]
mod versioned_chunk_db;

pub use chunk_db::*;
pub use key::*;

#[cfg(feature = "sled-snapshots")]
pub use versioned_chunk_db::*;

pub use sled;

use crate::dev_prelude::{ChunkKey, Compression};
use futures::future::join_all;
use sled::IVec;

async fn decompress_in_batches<N, Compr, F>(kvs: Vec<(IVec, IVec)>, mut chunk_rx: F)
where
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Compression,
    F: FnMut(ChunkKey<N>, Compr::Data),
{
    for batch in kvs.chunks(16) {
        for (chunk_key, chunk) in join_all(batch.iter().map(|(key, compressed_chunk)| async move {
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
