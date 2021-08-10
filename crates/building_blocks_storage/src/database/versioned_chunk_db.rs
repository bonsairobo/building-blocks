use super::{DatabaseKey, Delta, DeltaBatch, DeltaBatchBuilder, ReadableChunkDb};

use sled;

use crate::prelude::ChunkKey;

use sled::{transaction::TransactionResult, Transactional, Tree};
use sled_snapshots::{
    transactions::{create_child_snapshot, modify_current_leaf_snapshot, set_current_version},
    *,
};

impl<T> From<Delta<T, T>> for sled_snapshots::Delta<T> {
    fn from(d: Delta<T, T>) -> Self {
        match d {
            Delta::Insert(k, v) => sled_snapshots::Delta::Insert(k, v),
            Delta::Remove(k) => sled_snapshots::Delta::Remove(k),
        }
    }
}

/// The same as a [`ChunkDb`](super::ChunkDb) except it also supports versioning via the `sled-snapshots` crate.
///
/// There is some overhead incurred for writes and removals, which must rewrite deltas for the set of keys being accessed. Reads
/// do not incur any overhead.
pub struct VersionedChunkDb<N, Compr> {
    current_version: u64,
    data_tree: Tree,
    versions: VersionForest,
    deltas: DeltaMap,
    compression: Compr,
    marker: std::marker::PhantomData<N>,
}

/// A 2D `VersionedChunkDb`.
pub type VersionedChunkDb2<Compr> = VersionedChunkDb<[i32; 2], Compr>;
/// A 3D `VersionedChunkDb`.
pub type VersionedChunkDb3<Compr> = VersionedChunkDb<[i32; 3], Compr>;

impl<N, Compr> VersionedChunkDb<N, Compr> {
    pub fn new(
        current_version: u64,
        data_tree: Tree,
        versions: VersionForest,
        deltas: DeltaMap,
        compression: Compr,
    ) -> Self {
        Self {
            current_version,
            data_tree,
            versions,
            deltas,
            compression,
            marker: Default::default(),
        }
    }
}

impl<N, Compr> ReadableChunkDb for VersionedChunkDb<N, Compr> {
    type Compr = Compr;

    fn data_tree(&self) -> &sled::Tree {
        &self.data_tree
    }
}

impl<N, Compr> VersionedChunkDb<N, Compr>
where
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Copy,
{
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    pub fn versions(&self) -> &VersionForest {
        &self.versions
    }

    pub fn deltas(&self) -> &DeltaMap {
        &self.deltas
    }

    pub async fn flush(&self) -> sled::Result<usize> {
        self.data_tree.flush_async().await
    }

    pub fn start_delta_batch(
        &self,
    ) -> DeltaBatchBuilder<N, <ChunkKey<N> as DatabaseKey<N>>::OrdKey, Compr> {
        DeltaBatchBuilder::new(self.compression)
    }

    /// Applies a set of chunk deltas atomically. This will compress all of the inserted chunks asynchronously then insert them
    /// into the database.
    pub fn apply_deltas_to_current_version(&self, batch: DeltaBatch) -> TransactionResult<()> {
        let batch_deltas: Vec<_> = batch
            .deltas
            .into_iter()
            .map(|d| sled_snapshots::Delta::from(d))
            .collect();

        // Then atomically update the database.
        (&*self.versions, &*self.deltas, &self.data_tree).transaction(
            |(versions, deltas, data_tree)| {
                let versions = TransactionalVersionForest(versions);
                let deltas = TransactionalDeltaMap(deltas);
                modify_current_leaf_snapshot(
                    self.current_version,
                    versions,
                    deltas,
                    data_tree,
                    &batch_deltas,
                )
            },
        )
    }

    pub fn set_current_version(&mut self, target_version: u64) -> TransactionResult<()> {
        (&*self.versions, &*self.deltas, &self.data_tree).transaction(
            |(versions, deltas, data_tree)| {
                let versions = TransactionalVersionForest(versions);
                let deltas = TransactionalDeltaMap(deltas);
                set_current_version(
                    self.current_version,
                    target_version,
                    versions,
                    deltas,
                    data_tree,
                )?;
                Ok(())
            },
        )?;
        self.current_version = target_version;
        Ok(())
    }

    pub fn take_snapshot(&mut self) -> TransactionResult<u64> {
        let new_version = (&*self.versions, &*self.deltas).transaction(|(versions, deltas)| {
            let versions = TransactionalVersionForest(versions);
            let deltas = TransactionalDeltaMap(deltas);
            create_child_snapshot(self.current_version, true, versions, deltas)
        })?;
        self.current_version = new_version;
        Ok(new_version)
    }
}
