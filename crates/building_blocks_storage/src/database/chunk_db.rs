use super::{DatabaseKey, DeltaBatch, DeltaBatchBuilder, ReadableChunkDb};

use crate::prelude::ChunkKey;

use sled;
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
/// [`BincodeCompression`](crate::compression::BincodeCompression) if you absolutely need portability across machines with
/// different endianness.
pub struct ChunkDb<N, Compr = ()> {
    tree: Tree,
    compression: Compr,
    marker: std::marker::PhantomData<N>,
}

/// A 2D `ChunkDb`.
pub type ChunkDb2<Compr> = ChunkDb<[i32; 2], Compr>;
/// A 3D `ChunkDb`.
pub type ChunkDb3<Compr> = ChunkDb<[i32; 3], Compr>;

impl<N> ChunkDb<N> {
    /// Construct a `ChunkDb` without compression.
    pub fn new(tree: Tree) -> Self {
        Self {
            tree,
            compression: (),
            marker: Default::default(),
        }
    }
}

impl<N, Compr> ChunkDb<N, Compr> {
    /// Construct a `ChunkDb` with `compression`.
    pub fn new_with_compression(tree: Tree, compression: Compr) -> Self {
        Self {
            tree,
            compression,
            marker: Default::default(),
        }
    }
}

impl<N, Compr> ReadableChunkDb for ChunkDb<N, Compr> {
    type Compr = Compr;

    fn data_tree(&self) -> &Tree {
        &self.tree
    }
}

impl<N, Compr> ChunkDb<N, Compr>
where
    ChunkKey<N>: DatabaseKey<N>,
    Compr: Copy,
{
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

    use building_blocks_core::prelude::*;

    use sled::IVec;

    #[test]
    fn db_round_trip() -> sled::Result<()> {
        let chunk_mins = [
            PointN([16, 0, 0]),
            PointN([0, 16, 0]),
            PointN([0, 0, 16]),
            PointN([0, -16, 0]),
        ];
        let write_chunks: Vec<_> = chunk_mins
            .iter()
            .map(|&min| (ChunkKey3::new(0, min), IVec::from("data")))
            .collect();

        let db = sled::Config::default()
            .temporary(true)
            .use_compression(false)
            .mode(sled::Mode::LowSpace)
            .open()?;
        let tree = db.open_tree("chunks")?;

        let chunk_db = ChunkDb::new(tree);

        let mut batch = chunk_db.start_delta_batch();
        batch.add_raw_deltas(
            write_chunks
                .clone()
                .into_iter()
                .map(|(k, v)| Delta::Insert(k, v)),
        );
        chunk_db.apply_deltas(batch.build())?;

        // This octant should contain the chunks in the positive octant, but not the other chunk.
        let octant = Octant::new_unchecked(Point3i::ZERO, 32);

        let read_result = chunk_db.read_chunks_in_orthant(0, octant)?;
        let read_chunks: Vec<_> = read_result.take_with_raw_values().collect();

        let read_keys: Vec<_> = read_chunks.iter().map(|(k, _)| k.clone()).collect();
        let expected_read_keys: Vec<_> =
            [PointN([16, 0, 0]), PointN([0, 16, 0]), PointN([0, 0, 16])]
                .iter()
                .cloned()
                .map(|min| ChunkKey3::new(0, min))
                .collect();
        assert_eq!(read_keys, expected_read_keys);

        assert_eq!(&read_chunks, &write_chunks[0..3]);

        Ok(())
    }

    #[test]
    fn db_round_trip_with_compression() -> sled::Result<()> {
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
        let chunk_db = ChunkDb::new_with_compression(tree, compression);

        let mut batch = chunk_db.start_delta_batch();
        futures::executor::block_on(
            batch.add_and_compress_deltas(write_chunks.iter().map(|(k, v)| Delta::Insert(*k, v))),
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

        assert_eq!(&read_chunks, &write_chunks[0..3]);

        Ok(())
    }
}
