use super::{DatabaseKey, Delta, DeltaBatch, DeltaBatchBuilder, ReadResult};

pub use sled;

use super::key::map_bound;

use crate::prelude::ChunkKey;

use building_blocks_core::{orthants_covering_extent, prelude::*};

use core::ops::RangeBounds;
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

impl<N, Compr> VersionedChunkDb<N, Compr>
where
    PointN<N>: IntegerPoint<N>,
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Copy,
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

    /// Same as `ChunkDb::read_chunks_in_orthant`. Reads from the current version.
    pub fn read_chunks_in_orthant(
        &self,
        lod: u8,
        orthant: Orthant<N>,
    ) -> sled::Result<ReadResult<Compr>> {
        let range = ChunkKey::<N>::orthant_range(lod, orthant);
        self.read_morton_range(range)
    }

    /// Same as `ChunkDb::read_orthants_covering_extent`. Reads from the current version.
    pub fn read_orthants_covering_extent(
        &self,
        lod: u8,
        orthant_exponent: i32,
        extent: ExtentN<N>,
    ) -> sled::Result<ReadResult<Compr>> {
        // PERF: more parallelism?
        let mut result = ReadResult::default();
        for orthant in orthants_covering_extent(extent, orthant_exponent) {
            result.append(self.read_chunks_in_orthant(lod, orthant)?);
        }
        Ok(result)
    }

    /// Same as `ChunkDb::read_all_chunks`. Reads from the current version.
    pub fn read_all_chunks(&self, lod: u8) -> sled::Result<ReadResult<Compr>> {
        self.read_morton_range(ChunkKey::<N>::full_range(lod))
    }

    /// Same as `ChunkDb::read_morton_range`. Reads from the current version.
    pub fn read_morton_range<R>(&self, range: R) -> sled::Result<ReadResult<Compr>>
    where
        R: RangeBounds<<ChunkKey<N> as DatabaseKey<N>>::OrdKey>,
    {
        let key_range_start = map_bound(range.start_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_range_end = map_bound(range.end_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_value_pairs = self
            .data_tree
            .range((key_range_start, key_range_end))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReadResult::new(key_value_pairs))
    }
}
