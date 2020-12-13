use crate::array::ArrayN;

use building_blocks_core::prelude::*;

use core::ops::{Div, Mul};
use serde::{Deserialize, Serialize};

/// One piece of a chunked lattice map. Contains both some generic metadata and the data for each point in the chunk extent.
#[derive(Clone, Deserialize, Serialize)]
pub struct Chunk<N, T, M> {
    pub metadata: M,
    pub array: ArrayN<N, T>,
}

pub type Chunk2<T, M> = Chunk<[i32; 2], T, M>;
pub type Chunk3<T, M> = Chunk<[i32; 3], T, M>;

impl<N, T> Chunk<N, T, ()> {
    /// Constructs a chunk without metadata.
    pub fn with_array(array: ArrayN<N, T>) -> Self {
        Chunk {
            metadata: (),
            array,
        }
    }
}

pub trait ChunkShape<N> {
    /// Makes the mask required to convert points to chunk keys.
    fn mask(&self) -> PointN<N>;

    /// A chunk key is just the leading m bits of each component of a point, where m depends on the
    /// size of the chunk. It can also be interpreted as the minimum point of a chunk extent.
    fn chunk_key_containing_point(mask: &PointN<N>, p: &PointN<N>) -> PointN<N>;

    fn ilog2(&self) -> PointN<N>;
}

macro_rules! impl_chunk_shape {
    ($point:ty, $dims:ty) => {
        impl ChunkShape<$dims> for $point {
            fn mask(&self) -> $point {
                assert!(self.dimensions_are_powers_of_2());

                self.map_components_unary(|c| !(c - 1))
            }

            fn chunk_key_containing_point(mask: &$point, p: &$point) -> $point {
                mask.map_components_binary(p, |c1, c2| c1 & c2)
            }

            fn ilog2(&self) -> $point {
                self.map_components_unary(|c| c.trailing_zeros() as i32)
            }
        }
    };
}

impl_chunk_shape!(Point2i, [i32; 2]);
impl_chunk_shape!(Point3i, [i32; 3]);

/// Translates from lattice coordinates to chunk key space.
///
/// The key for a chunk is the minimum point of that chunk's extent.
#[derive(Clone, Copy, Debug)]
pub struct ChunkIndexer<N> {
    chunk_shape: PointN<N>,
    chunk_shape_mask: PointN<N>,
    chunk_shape_log2: PointN<N>,
}

impl<N> ChunkIndexer<N>
where
    PointN<N>: ChunkShape<N> + IntegerPoint,
    ExtentN<N>: IntegerExtent<N>,
{
    pub fn new(chunk_shape: PointN<N>) -> Self {
        Self {
            chunk_shape,
            chunk_shape_mask: chunk_shape.mask(),
            chunk_shape_log2: chunk_shape.ilog2(),
        }
    }

    /// Determines whether `key` is a valid chunk key. This means it must be a multiple of the chunk shape.
    pub fn chunk_key_is_valid(&self, key: &PointN<N>) -> bool {
        self.chunk_shape.mul(key.div(self.chunk_shape)).eq(key)
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
    pub fn chunk_key_containing_point(&self, point: &PointN<N>) -> PointN<N> {
        PointN::chunk_key_containing_point(&self.chunk_shape_mask(), point)
    }

    /// Returns an iterator over all chunk keys for chunks that overlap the given extent.
    pub fn chunk_keys_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        let key_min = extent.minimum.vector_right_shift(&self.chunk_shape_log2);
        let key_max = extent.max().vector_right_shift(&self.chunk_shape_log2);
        let shape_log2 = self.chunk_shape_log2;

        ExtentN::from_min_and_max(key_min, key_max)
            .iter_points()
            .map(move |p| p.vector_left_shift(&shape_log2))
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
}
