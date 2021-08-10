use crate::dev_prelude::{ChunkKey, Compression, DatabaseKey, Delta};

use futures::future::join_all;
use sled::IVec;
use std::borrow::Borrow;

/// Creates a [DeltaBatch]. This handles sorting the deltas in Morton order and compressing the chunk data.
pub struct DeltaBatchBuilder<N, K, Compr> {
    compressed_deltas: Vec<Delta<K, IVec>>,
    compression: Compr,
    marker: std::marker::PhantomData<N>,
}

impl<N, K, Compr> DeltaBatchBuilder<N, K, Compr> {
    pub fn new(compression: Compr) -> Self {
        Self {
            compressed_deltas: Default::default(),
            compression,
            marker: Default::default(),
        }
    }
}

impl<N, K, Compr> DeltaBatchBuilder<N, K, Compr>
where
    Compr: Compression + Copy,
    ChunkKey<N>: DatabaseKey<N, OrdKey = K>,
{
    pub fn add_compressed_deltas(
        &mut self,
        deltas: impl Iterator<Item = Delta<ChunkKey<N>, IVec>>,
    ) {
        self.compressed_deltas
            .extend(deltas.map(|delta| match delta {
                Delta::Insert(k, v) => Delta::Insert(ChunkKey::<N>::into_ord_key(k), v),
                Delta::Remove(k) => Delta::Remove(ChunkKey::<N>::into_ord_key(k)),
            }));
    }

    /// Compresses `deltas` concurrently and adds them to the batch.
    pub async fn add_deltas<Data>(&mut self, deltas: impl Iterator<Item = Delta<ChunkKey<N>, Data>>)
    where
        Data: Borrow<Compr::Data>,
    {
        // Compress all of the chunks in parallel.
        let compression = self.compression.clone();
        let mut compressed_deltas: Vec<_> = join_all(deltas.map(|delta| async move {
            match delta {
                Delta::Insert(k, v) => Delta::Insert(
                    ChunkKey::<N>::into_ord_key(k),
                    // PERF: IVec will copy the bytes instead of moving, because it needs to also allocate room for an internal
                    // header
                    IVec::from(compression.compress(v.borrow()).take_bytes()),
                ),
                Delta::Remove(k) => Delta::Remove(ChunkKey::<N>::into_ord_key(k)),
            }
        }))
        .await;
        self.compressed_deltas.append(&mut compressed_deltas);
    }

    /// Sorts the deltas by Morton key and converts them to `IVec` key-value pairs for `sled`.
    pub fn build(mut self) -> DeltaBatch
    where
        K: Copy + Ord,
    {
        // Sort them by the Ord key.
        self.compressed_deltas.sort_by_key(|delta| *delta.key());

        let deltas: Vec<_> = self
            .compressed_deltas
            .into_iter()
            .map(|delta| match delta {
                Delta::Insert(k, v) => Delta::Insert(
                    IVec::from(ChunkKey::<N>::ord_key_to_be_bytes(k).as_ref()),
                    v,
                ),
                Delta::Remove(k) => {
                    Delta::Remove(IVec::from(ChunkKey::<N>::ord_key_to_be_bytes(k).as_ref()))
                }
            })
            .collect();

        DeltaBatch { deltas }
    }
}

/// A set of [Delta]s to be atomically applied to a [ChunkDb](super::ChunkDb) or [VersionedChunkDb](super::VersionedChunkDb).
///
/// Can be created with a [DeltaBatchBuilder].
#[derive(Default)]
pub struct DeltaBatch {
    pub(crate) deltas: Vec<Delta<IVec, IVec>>,
}

impl From<DeltaBatch> for sled::Batch {
    fn from(batch: DeltaBatch) -> Self {
        let mut new_batch = sled::Batch::default();
        for delta in batch.deltas.into_iter() {
            match delta {
                Delta::Insert(key_bytes, chunk_bytes) => {
                    new_batch.insert(key_bytes.as_ref(), chunk_bytes);
                }
                Delta::Remove(key_bytes) => new_batch.remove(key_bytes.as_ref()),
            }
        }
        new_batch
    }
}
