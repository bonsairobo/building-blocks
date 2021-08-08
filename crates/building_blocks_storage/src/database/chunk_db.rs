use super::{key::DatabaseKey, Delta, DeltaBatchBuilder};

pub use sled;

use super::decompress_in_batches;

use crate::prelude::{ChunkKey, Compression};

use building_blocks_core::{orthants_covering_extent, prelude::*};

use core::ops::RangeBounds;
use sled::Tree;
use std::borrow::Borrow;

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

    /// Applies a set of chunk deltas atomically. This will compress all of the inserted chunks asynchronously then insert them
    /// into the database.
    pub async fn update<Data>(
        &self,
        deltas: impl Iterator<Item = Delta<ChunkKey<N>, Data>>,
    ) -> sled::Result<()>
    where
        Data: Borrow<Compr::Data>,
    {
        let mut builder = DeltaBatchBuilder::default();
        builder.add_deltas(self.compression, deltas).await;
        let batch = sled::Batch::from(builder.build());
        self.tree.apply_batch(batch)
    }

    /// Scans the given orthant for chunks, decompresses them, then passes them to `chunk_rx`. Because chunk keys are stored in
    /// Morton order, the chunks in any orthant are guaranteed to be contiguous.
    ///
    /// The `orthant` is expected in voxel units, not chunk units.
    pub async fn read_chunks_in_orthant(
        &self,
        lod: u8,
        orthant: Orthant<N>,
        chunk_rx: impl FnMut(ChunkKey<N>, Compr::Data),
    ) -> sled::Result<()> {
        let range = ChunkKey::<N>::orthant_range(lod, orthant);

        self.read_range(range, chunk_rx).await
    }

    /// This is like `read_chunks_in_orthant`, but it works for the given `extent`. Since Morton order only guarantees
    /// contiguity within a single `Orthant`, we should not naively scan from the Morton of `extent.minimum` to `extent.max()`.
    /// Rather, we scan a set of `Orthant`s that covers `extent`. This covering is *at least* sufficient to cover the extent,
    /// and it gets more exact as `orthant_exponent` (log2 of the side length) gets smaller. However, for exactness, you must
    /// necessarily do more scans.
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

    /// Reads all chunks in the given `lod`, passing them to `chunk_rx`.
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
        let read_kvs = self.tree.range(range).collect::<Result<Vec<_>, _>>()?;
        decompress_in_batches::<_, Compr, _>(read_kvs, chunk_rx).await;

        Ok(())
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
    use crate::prelude::{Array3x2, ChunkKey3, FastArrayCompressionNx2, FromBytesCompression, Lz4};

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

        futures::executor::block_on(
            chunk_db.update(write_chunks.iter().map(|(k, v)| Delta::Insert(*k, v))),
        )?;

        // This octant should contain the chunks in the positive octant, but not the other chunk.
        let octant = Octant::new_unchecked(Point3i::ZERO, 32);

        let mut read_chunks = Vec::new();
        futures::executor::block_on(
            chunk_db.read_chunks_in_orthant(0, octant, |k, v| read_chunks.push((k, v))),
        )?;

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
