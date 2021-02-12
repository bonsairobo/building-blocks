use crate::array::ArrayN;

use building_blocks_core::prelude::*;

use core::ops::{Div, Mul};
use serde::{Deserialize, Serialize};

/// One piece of a chunked lattice map. Contains both some generic metadata and the data for each point in the chunk extent.
#[derive(Clone, Deserialize, Serialize)]
pub struct Chunk<N, T, Meta = ()> {
    pub metadata: Meta,
    pub array: ArrayN<N, T>,
}

/// A 2-dimensional `Chunk`.
pub type Chunk2<T, Meta = ()> = Chunk<[i32; 2], T, Meta>;
/// A 3-dimensional `Chunk`.
pub type Chunk3<T, Meta = ()> = Chunk<[i32; 3], T, Meta>;

impl<N, T> Chunk<N, T, ()> {
    /// Constructs a chunk without metadata.
    pub fn with_array(array: ArrayN<N, T>) -> Self {
        Chunk {
            metadata: (),
            array,
        }
    }
}

/// Translates from lattice coordinates to chunk key space.
///
/// The key for a chunk is the minimum point of that chunk's extent.
#[derive(Clone, Debug)]
pub struct ChunkIndexer<N> {
    chunk_shape: PointN<N>,
    chunk_shape_mask: PointN<N>,
    chunk_shape_log2: PointN<N>,
}

impl<N> ChunkIndexer<N>
where
    PointN<N>: IntegerPoint<N>,
{
    pub fn new(chunk_shape: PointN<N>) -> Self {
        assert!(chunk_shape.dimensions_are_powers_of_2());

        Self {
            chunk_shape,
            chunk_shape_mask: !(chunk_shape - PointN::ONES),
            chunk_shape_log2: chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32),
        }
    }

    /// Determines whether `key` is a valid chunk key. This means it must be a multiple of the chunk shape.
    pub fn chunk_key_is_valid(&self, key: PointN<N>) -> bool {
        self.chunk_shape.mul(key.div(self.chunk_shape)).eq(&key)
    }

    /// The constant shape of a chunk. The same for all chunks.
    pub fn chunk_shape(&self) -> PointN<N> {
        self.chunk_shape
    }

    /// The mask used for calculating the chunk key of a chunk that contains a given point.
    pub fn chunk_shape_mask(&self) -> PointN<N> {
        self.chunk_shape_mask
    }

    /// Returns the key of the chunk that contains `point`.
    pub fn chunk_key_containing_point(&self, point: PointN<N>) -> PointN<N> {
        self.chunk_shape_mask() & point
    }

    /// Returns an iterator over all chunk keys for chunks that overlap the given extent.
    pub fn chunk_keys_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        let key_min = extent.minimum >> self.chunk_shape_log2;
        let key_max = extent.max() >> self.chunk_shape_log2;
        let shape_log2 = self.chunk_shape_log2;

        ExtentN::from_min_and_max(key_min, key_max)
            .iter_points()
            .map(move |p| p << shape_log2)
    }

    /// The extent spanned by the chunk at `key`.
    pub fn extent_for_chunk_at_key(&self, key: PointN<N>) -> ExtentN<N> {
        ExtentN::from_min_and_shape(key, self.chunk_shape)
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use building_blocks_core::Extent3i;

    #[test]
    fn chunk_keys_for_extent_gives_keys_for_chunks_overlapping_extent() {
        let indexer = ChunkIndexer::new(PointN([16; 3]));
        let query_extent = Extent3i::from_min_and_shape(PointN([15; 3]), PointN([16; 3]));
        let chunk_keys: Vec<_> = indexer.chunk_keys_for_extent(&query_extent).collect();

        assert_eq!(
            chunk_keys,
            vec![
                PointN([0, 0, 0]),
                PointN([16, 0, 0]),
                PointN([0, 16, 0]),
                PointN([16, 16, 0]),
                PointN([0, 0, 16]),
                PointN([16, 0, 16]),
                PointN([0, 16, 16]),
                PointN([16, 16, 16])
            ]
        );
    }

    #[test]
    fn chunk_key_for_negative_point_is_negative() {
        let indexer = ChunkIndexer::new(PointN([16; 3]));
        let p = PointN([-1; 3]);
        let key = indexer.chunk_key_containing_point(p);
        assert_eq!(key, PointN([-16; 3]));
    }
}
