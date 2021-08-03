use super::{key::DatabaseKey, prepare_deltas_for_update, Delta};

pub use sled;

use super::decompress_in_batches;

use crate::prelude::{ChunkKey, Compression};

use building_blocks_core::{orthants_covering_extent, prelude::*};

use core::ops::RangeBounds;
use sled::{transaction::TransactionResult, Transactional, Tree};
use sled_snapshots::{
    transactions::{create_child_snapshot, modify_current_leaf_snapshot, set_current_version},
    *,
};
use std::borrow::Borrow;

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

impl<N, Compr> VersionedChunkDb<N, Compr>
where
    PointN<N>: IntegerPoint<N>,
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Compression + Copy,
{
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    pub fn data_tree(&self) -> &Tree {
        &self.data_tree
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

    /// Applies a set of deltas. This will compress all of the inserted chunks asynchronously then insert them into the
    /// database.
    pub async fn update_current_version<Data>(
        &self,
        deltas: impl Iterator<Item = Delta<ChunkKey<N>, Data>>,
    ) -> TransactionResult<()>
    where
        Data: Borrow<Compr::Data>,
    {
        // Compress the chunks asynchronously.
        let compressed_deltas: Vec<_> = prepare_deltas_for_update(self.compression, deltas)
            .await
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
                    &compressed_deltas,
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

    /// Same as `ChunkDb::read_chunks_in_orthant`. Reads from the current version.
    pub async fn read_chunks_in_orthant(
        &self,
        lod: u8,
        orthant: Orthant<N>,
        chunk_rx: impl FnMut(ChunkKey<N>, Compr::Data),
    ) -> sled::Result<()> {
        let range = ChunkKey::<N>::orthant_range(lod, orthant);
        self.read_range(range, chunk_rx).await
    }

    /// Same as `ChunkDb::read_orthants_covering_extent`. Reads from the current version.
    pub async fn read_orthants_covering_extent(
        &self,
        lod: u8,
        orthant_exponent: i32,
        extent: ExtentN<N>,
        mut chunk_rx: impl FnMut(ChunkKey<N>, Compr::Data),
    ) -> sled::Result<()> {
        // PERF: more parallelism?
        for orthant in orthants_covering_extent(extent, orthant_exponent) {
            self.read_chunks_in_orthant(lod, orthant, &mut chunk_rx)
                .await?;
        }
        Ok(())
    }

    /// Same as `ChunkDb::read_all_chunks`. Reads from the current version.
    pub async fn read_all_chunks(
        &self,
        lod: u8,
        chunk_rx: impl FnMut(ChunkKey<N>, Compr::Data),
    ) -> sled::Result<()> {
        self.read_range(ChunkKey::<N>::full_range(lod), chunk_rx)
            .await
    }

    async fn read_range<R>(
        &self,
        range: R,
        chunk_rx: impl FnMut(ChunkKey<N>, Compr::Data),
    ) -> sled::Result<()>
    where
        R: RangeBounds<<ChunkKey<N> as DatabaseKey<N>>::KeyBytes>,
    {
        let read_kvs = self.data_tree.range(range).collect::<Result<Vec<_>, _>>()?;
        decompress_in_batches::<_, Compr, _>(read_kvs, chunk_rx).await;
        Ok(())
    }
}
