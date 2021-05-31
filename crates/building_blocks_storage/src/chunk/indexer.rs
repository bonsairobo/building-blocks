use building_blocks_core::prelude::*;

use core::ops::{Div, Mul};
use serde::{Deserialize, Serialize};

/// Uses a bitmask to calculate the minimum of the chunk that contains a given point.
///
/// We use chunk minimums as keys for chunk storage.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
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

    /// Determines whether `min` is a valid chunk minimum. This means it must be a multiple of the chunk shape.
    pub fn chunk_min_is_valid(&self, min: PointN<N>) -> bool {
        self.chunk_shape.mul(min.div(self.chunk_shape)).eq(&min)
    }

    /// The constant shape of a chunk. The same for all chunks.
    pub fn chunk_shape(&self) -> PointN<N> {
        self.chunk_shape
    }

    /// The mask used for calculating the minimum of a chunk that contains a given point.
    pub fn chunk_shape_mask(&self) -> PointN<N> {
        self.chunk_shape_mask
    }

    /// Returns the minimum of the chunk that contains `point`.
    pub fn min_of_chunk_containing_point(&self, point: PointN<N>) -> PointN<N> {
        self.chunk_shape_mask() & point
    }

    /// Returns an iterator over all chunk minimums for chunks that overlap the given extent.
    pub fn chunk_mins_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        let range_min = extent.minimum >> self.chunk_shape_log2;
        let range_max = extent.max() >> self.chunk_shape_log2;
        let shape_log2 = self.chunk_shape_log2;

        ExtentN::from_min_and_max(range_min, range_max)
            .iter_points()
            .map(move |p| p << shape_log2)
    }

    /// The extent spanned by the chunk at `min`.
    pub fn extent_for_chunk_with_min(&self, min: PointN<N>) -> ExtentN<N> {
        ExtentN::from_min_and_shape(min, self.chunk_shape)
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
    fn chunk_mins_for_extent_gives_mins_for_chunks_overlapping_extent() {
        let indexer = ChunkIndexer::new(Point3i::fill(16));
        let query_extent = Extent3i::from_min_and_shape(Point3i::fill(15), Point3i::fill(16));
        let chunk_mins: Vec<_> = indexer.chunk_mins_for_extent(&query_extent).collect();

        assert_eq!(
            chunk_mins,
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
    fn chunk_min_for_negative_point_is_negative() {
        let indexer = ChunkIndexer::new(Point3i::fill(16));
        let p = Point3i::fill(-1);
        let min = indexer.min_of_chunk_containing_point(p);
        assert_eq!(min, Point3i::fill(-16));
    }
}
