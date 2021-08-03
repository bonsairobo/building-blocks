mod chunk_db;
mod key;

#[cfg(feature = "sled-snapshots")]
mod versioned_chunk_db;

pub use chunk_db::*;
pub use key::*;

#[cfg(feature = "sled-snapshots")]
pub use versioned_chunk_db::*;

pub use sled;

pub enum Delta<K, V> {
    Insert(K, V),
    Remove(K),
}

impl<K, V> Delta<K, V> {
    fn key(&self) -> &K {
        match self {
            Self::Insert(k, _) => &k,
            Self::Remove(k) => &k,
        }
    }
}

use crate::dev_prelude::{ChunkKey, Compression};
use futures::future::join_all;
use sled::IVec;
use std::borrow::Borrow;

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

async fn prepare_deltas_for_update<N, Compr, Data>(
    compression: Compr,
    deltas: impl Iterator<Item = Delta<ChunkKey<N>, Data>>,
) -> impl Iterator<Item = Delta<IVec, IVec>>
where
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Compression + Copy,
    Data: Borrow<Compr::Data>,
{
    // First compress all of the chunks in parallel.
    let mut compressed_chunks: Vec<_> = join_all(deltas.map(|delta| async move {
        match delta {
            Delta::Insert(k, v) => Delta::Insert(
                ChunkKey::<N>::into_ord_key(k),
                compression.compress(v.borrow()),
            ),
            Delta::Remove(k) => Delta::Remove(ChunkKey::<N>::into_ord_key(k)),
        }
    }))
    .await
    .into_iter()
    .collect();
    // Sort them by the Ord key.
    compressed_chunks.sort_by_key(|delta| *delta.key());

    compressed_chunks.into_iter().map(|delta| {
        // PERF: IVec will copy the bytes instead of moving, because it needs to also allocate room for an internal header
        match delta {
            Delta::Insert(k, v) => Delta::Insert(
                IVec::from(ChunkKey::<N>::ord_key_to_be_bytes(k).as_ref()),
                IVec::from(v.take_bytes()),
            ),
            Delta::Remove(k) => {
                Delta::Remove(IVec::from(ChunkKey::<N>::ord_key_to_be_bytes(k).as_ref()))
            }
        }
    })
}
