use super::{key::map_bound, DatabaseKey, DeltaBatch, DeltaBatchBuilder, ReadResult};

pub use sled;

use crate::prelude::{ChunkKey, Compression};

use building_blocks_core::{orthants_covering_extent, prelude::*};

use core::ops::RangeBounds;
use sled::Tree;

/// A persistent, crash-consistent key-value store of compressed chunks, backed by the `sled` crate.
///
/// The keys are Morton codes for the corresponding chunk coordinates. This ensures that all of the chunks in an orthant are
/// stored in a contiguous key space.
///
/// Note that while writes are applied atomically, reads are not isolated. Reads rely on iteration over Morton keys, and `sled`
/// does not yet provide transactional iteration.
///
/// The DB values are only portable if the `compression` used respects endianness of the current machine. Use
/// `BincodeCompression` if you absolutely need portability across machines with different endianness.
pub struct ChunkDb<N, Compr> {
    tree: Tree,
    compression: Compr,
    marker: std::marker::PhantomData<N>,
}

/// A 2D `ChunkDb`.
pub type ChunkDb2<Compr> = ChunkDb<[i32; 2], Compr>;
/// A 3D `ChunkDb`.
pub type ChunkDb3<Compr> = ChunkDb<[i32; 3], Compr>;

impl<N, Compr> ChunkDb<N, Compr> {
    pub fn new(tree: Tree, compression: Compr) -> Self {
        Self {
            tree,
            compression,
            marker: Default::default(),
        }
    }
}

impl<N, Compr> ChunkDb<N, Compr>
where
    PointN<N>: IntegerPoint<N>,
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Compression + Copy,
{
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    pub async fn flush(&self) -> sled::Result<usize> {
        self.tree.flush_async().await
    }

    pub fn start_delta_batch(
        &self,
    ) -> DeltaBatchBuilder<N, <ChunkKey<N> as DatabaseKey<N>>::OrdKey, Compr> {
        DeltaBatchBuilder::new(self.compression)
    }

    /// Applies a set of chunk deltas atomically.
    pub fn apply_deltas(&self, batch: DeltaBatch) -> sled::Result<()> {
        self.tree.apply_batch(sled::Batch::from(batch))
    }

    /// Scans the given orthant for chunks. Because chunk keys are stored in Morton order, the chunks in any orthant are
    /// guaranteed to be contiguous.
    ///
    /// The `orthant` is expected in voxel units, not chunk units.
    pub fn read_chunks_in_orthant(
        &self,
        lod: u8,
        orthant: Orthant<N>,
    ) -> sled::Result<ReadResult<Compr>> {
        let range = ChunkKey::<N>::orthant_range(lod, orthant);
        self.read_morton_range(range)
    }

    /// This is like `read_chunks_in_orthant`, but it works for the given `extent`. Since Morton order only guarantees
    /// contiguity within a single `Orthant`, we should not naively scan from the Morton of `extent.minimum` to `extent.max()`.
    /// Rather, we scan a set of `Orthant`s that covers `extent`. This covering is *at least* sufficient to cover the extent,
    /// and it gets more exact as `orthant_exponent` (log2 of the side length) gets smaller. However, for exactness, you must
    /// necessarily do more scans.
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

    /// Reads all chunks in the given `lod`.
    pub fn read_all_chunks(&self, lod: u8) -> sled::Result<ReadResult<Compr>> {
        self.read_morton_range(ChunkKey::<N>::full_range(lod))
    }

    /// Reads all chunks in the given `range` of Morton codes.
    pub fn read_morton_range<R>(&self, range: R) -> sled::Result<ReadResult<Compr>>
    where
        R: RangeBounds<<ChunkKey<N> as DatabaseKey<N>>::OrdKey>,
    {
        let key_range_start = map_bound(range.start_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_range_end = map_bound(range.end_bound(), |k| ChunkKey::ord_key_to_be_bytes(*k));
        let key_value_pairs = self
            .tree
            .range((key_range_start, key_range_end))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReadResult::new(key_value_pairs))
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use crate::{
        database::Delta,
        prelude::{Array3x2, ChunkKey3, FastArrayCompressionNx2, FromBytesCompression, Lz4},
    };

    use super::*;

    #[test]
    fn db_round_trip() -> sled::Result<()> {
        let chunk_mins = [
            PointN([16, 0, 0]),
            PointN([0, 16, 0]),
            PointN([0, 0, 16]),
            PointN([0, -16, 0]),
        ];
        let chunk_shape = Point3i::fill(16);
        let write_chunks: Vec<_> = chunk_mins
            .iter()
            .map(|&min| {
                (
                    ChunkKey3::new(0, min),
                    Array3x2::fill(Extent3i::from_min_and_shape(min, chunk_shape), (1u16, b'a')),
                )
            })
            .collect();

        let db = sled::Config::default()
            .temporary(true)
            .use_compression(false)
            .mode(sled::Mode::LowSpace)
            .open()?;
        let tree = db.open_tree("chunks")?;

        // NOTE: This compression is not portable because it is naive to endianness.
        let compression = FastArrayCompressionNx2::from_bytes_compression(Lz4 { level: 10 });
        let chunk_db = ChunkDb::new(tree, compression);

        let mut batch = chunk_db.start_delta_batch();
        futures::executor::block_on(
            batch.add_deltas(write_chunks.iter().map(|(k, v)| Delta::Insert(*k, v))),
        );
        chunk_db.apply_deltas(batch.build())?;

        // This octant should contain the chunks in the positive octant, but not the other chunk.
        let octant = Octant::new_unchecked(Point3i::ZERO, 32);

        let read_result = chunk_db.read_chunks_in_orthant(0, octant)?;
        let mut read_chunks = Vec::new();
        futures::executor::block_on(read_result.decompress(|k, v| read_chunks.push((k, v))));

        let read_keys: Vec<_> = read_chunks.iter().map(|(k, _)| k.clone()).collect();
        let expected_read_keys: Vec<_> =
            [PointN([16, 0, 0]), PointN([0, 16, 0]), PointN([0, 0, 16])]
                .iter()
                .cloned()
                .map(|min| ChunkKey3::new(0, min))
                .collect();
        assert_eq!(read_keys, expected_read_keys);

        assert_eq!(
            read_chunks,
            vec![
                write_chunks[0].clone(),
                write_chunks[1].clone(),
                write_chunks[2].clone()
            ]
        );

        Ok(())
    }
}
