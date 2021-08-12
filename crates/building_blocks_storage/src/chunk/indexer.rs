use crate::dev_prelude::Local;

use building_blocks_core::prelude::*;

use core::ops::{Div, Mul};
use serde::{Deserialize, Serialize};

/// Calculates chunk locations, e.g. minimums and downsampling destinations.
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
    #[inline]
    pub fn new(chunk_shape: PointN<N>) -> Self {
        assert!(chunk_shape.dimensions_are_powers_of_2());

        Self {
            chunk_shape,
            chunk_shape_mask: !(chunk_shape - PointN::ONES),
            chunk_shape_log2: chunk_shape.map_components_unary(|c| c.trailing_zeros() as i32),
        }
    }

    /// Determines whether `min` is a valid chunk minimum. This means it must be a multiple of the chunk shape.
    #[inline]
    pub fn chunk_min_is_valid(&self, min: PointN<N>) -> bool {
        self.chunk_shape.mul(min.div(self.chunk_shape)).eq(&min)
    }

    /// The constant shape of a chunk. The same for all chunks.
    #[inline]
    pub fn chunk_shape(&self) -> PointN<N> {
        self.chunk_shape
    }

    /// Returns the minimum of the chunk that contains `point`.
    #[inline]
    pub fn min_of_chunk_containing_point(&self, point: PointN<N>) -> PointN<N> {
        self.chunk_shape_mask & point
    }

    /// Returns an iterator over all chunk minimums for chunks that overlap the given extent.
    #[inline]
    pub fn chunk_mins_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        let range_min = extent.minimum >> self.chunk_shape_log2;
        let range_max = extent.max() >> self.chunk_shape_log2;
        let shape_log2 = self.chunk_shape_log2;

        ExtentN::from_min_and_max(range_min, range_max)
            .iter_points()
            .map(move |p| p << shape_log2)
    }

    /// The extent spanned by the chunk at `min`.
    #[inline]
    pub fn extent_for_chunk_with_min(&self, min: PointN<N>) -> ExtentN<N> {
        ExtentN::from_min_and_shape(min, self.chunk_shape)
    }

    /// Given a chunk at `chunk_min`, returns the minimum of the parent chunk.
    #[inline]
    pub fn parent_chunk_min(&self, chunk_min: PointN<N>) -> PointN<N> {
        (chunk_min & (self.chunk_shape_mask << 1)) >> 1
    }

    /// Given a chunk at `chunk_min`, returns the minimum of the child chunk with `child_index`.
    #[inline]
    pub fn child_chunk_min(&self, chunk_min: PointN<N>, child_index: u8) -> PointN<N> {
        (chunk_min << 1) + (PointN::corner_offset(child_index) << self.chunk_shape_log2)
    }

    /// When downsampling a chunk at level `L`, the samples are used at the returned destination within level `L + 1`.
    #[inline]
    pub fn downsample_destination(&self, src_chunk_min: PointN<N>) -> DownsampleDestination<N> {
        let double_mask = self.chunk_shape_mask << 1;
        let double_parent_min = src_chunk_min & double_mask;
        let dst_chunk_min = double_parent_min >> 1;
        let dst_offset = Local((src_chunk_min - double_parent_min) >> 1);
        DownsampleDestination {
            dst_chunk_min,
            dst_offset,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownsampleDestination<N> {
    pub dst_chunk_min: PointN<N>,
    pub dst_offset: Local<N>,
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

    #[test]
    fn parent_chunk_min() {
        let indexer = ChunkIndexer::new(Point3i::fill(4));
        let lod0_p = Point3i::fill(15);
        let lod0_min = indexer.min_of_chunk_containing_point(lod0_p);
        assert_eq!(lod0_min, Point3i::fill(12));
        let lod1_min = indexer.parent_chunk_min(lod0_min);
        assert_eq!(lod1_min, Point3i::fill(4));
        let lod2_min = indexer.parent_chunk_min(lod1_min);
        assert_eq!(lod2_min, Point3i::fill(0));
    }

    #[test]
    fn child_chunk_min() {
        let indexer = ChunkIndexer::new(Point3i::fill(4));
        let lod2_min = Point3i::fill(0);
        let lod1_min = indexer.child_chunk_min(lod2_min, 0b111);
        assert_eq!(lod1_min, Point3i::fill(4));
        let lod0_min = indexer.child_chunk_min(lod1_min, 0b111);
        assert_eq!(lod0_min, Point3i::fill(12));
    }

    #[test]
    fn downsample_destination() {
        let chunk_shape = Point3i::fill(16);
        let indexer = ChunkIndexer::new(chunk_shape);

        let src_key = chunk_shape;
        let dst = indexer.downsample_destination(src_key);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_min: Point3i::ZERO,
                dst_offset: Local(chunk_shape / 2),
            }
        );

        let src_key = 2 * chunk_shape;
        let dst = indexer.downsample_destination(src_key);
        assert_eq!(
            dst,
            DownsampleDestination {
                dst_chunk_min: chunk_shape,
                dst_offset: Local(Point3i::ZERO),
            }
        );
    }
}
