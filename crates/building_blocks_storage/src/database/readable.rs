use super::{key::map_bound, DatabaseKey, ReadResult};

use crate::prelude::ChunkKey;

use building_blocks_core::{orthants_covering_extent, prelude::*};

use core::ops::RangeBounds;

/// Shared behavior for chunk databases, i.e. those that are keyed on `ChunkKey`.
pub trait ReadableChunkDb {
    type Compr;

    fn data_tree(&self) -> &sled::Tree;

    /// Scans the given orthant for chunks. Because chunk keys are stored in Morton order, the chunks in any orthant are
    /// guaranteed to be contiguous.
    ///
    /// The `orthant` is expected in voxel units, not chunk units.
    fn read_chunks_in_orthant<N>(
        &self,
        lod: u8,
        orthant: Orthant<N>,
    ) -> sled::Result<ReadResult<Self::Compr>>
    where
        ChunkKey<N>: DatabaseKey<N>,
    {
        let range = ChunkKey::<N>::orthant_range(lod, orthant);
        self.read_morton_range(range)
    }

    /// This is like `read_chunks_in_orthant`, but it works for the given `extent`. Since Morton order only guarantees
    /// contiguity within a single `Orthant`, we should not naively scan from the Morton of `extent.minimum` to `extent.max()`.
    /// Rather, we scan a set of `Orthant`s that covers `extent`. This covering is *at least* sufficient to cover the extent,
    /// and it gets more exact as `orthant_exponent` (log2 of the side length) gets smaller. However, for exactness, you must
    /// necessarily do more scans.
    fn read_orthants_covering_extent<N>(
        &self,
        lod: u8,
        orthant_exponent: i32,
        extent: ExtentN<N>,
    ) -> sled::Result<ReadResult<Self::Compr>>
    where
        PointN<N>: IntegerPoint,
        ChunkKey<N>: DatabaseKey<N>,
    {
        // PERF: more parallelism?
        let mut result = ReadResult::default();
        for orthant in orthants_covering_extent(extent, orthant_exponent) {
            result.append(self.read_chunks_in_orthant(lod, orthant)?);
        }
        Ok(result)
    }

    /// Reads all chunks in the given `lod`.
    fn read_all_chunks<N>(&self, lod: u8) -> sled::Result<ReadResult<Self::Compr>>
    where
        ChunkKey<N>: DatabaseKey<N>,
    {
        self.read_morton_range(ChunkKey::<N>::full_range(lod))
    }

    /// Reads all chunks in the given `range` of Morton codes.
    fn read_morton_range<N, R>(&self, range: R) -> sled::Result<ReadResult<Self::Compr>>
    where
        ChunkKey<N>: DatabaseKey<N>,
        R: RangeBounds<<ChunkKey<N> as DatabaseKey<N>>::OrdKey>,
    {
        let key_range_start = map_bound(range.start_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_range_end = map_bound(range.end_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_value_pairs = self
            .data_tree()
            .range((key_range_start, key_range_end))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReadResult::new(key_value_pairs))
    }
}
