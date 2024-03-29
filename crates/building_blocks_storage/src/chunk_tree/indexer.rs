use crate::dev_prelude::Local;

use building_blocks_core::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Arbitrarily chosen. Certainly can't exceed the number of bits in an i32.
pub const MAX_LODS: usize = 20;

/// Calculates chunk locations, e.g. minimums and downsampling destinations.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChunkIndexer<N> {
    chunk_shape: PointN<N>,
    chunk_shape_mask: PointN<N>,
    chunk_shape_log2: PointN<N>,
}

/// A 2-dimensional [`ChunkIndexer`].
pub type ChunkIndexer2 = ChunkIndexer<[i32; 2]>;
/// A 3-dimensional [`ChunkIndexer`].
pub type ChunkIndexer3 = ChunkIndexer<[i32; 3]>;

impl<N> ChunkIndexer<N>
where
    PointN<N>: IntegerPoint,
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
        ((!self.chunk_shape_mask) & min) == PointN::ZERO
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

    /// Returns an iterator over all chunk minimums for chunks that overlap the given extent. Only applies to the LOD in which
    /// `extent` resides.
    #[inline]
    pub fn chunk_mins_for_extent(&self, extent: &ExtentN<N>) -> impl Iterator<Item = PointN<N>> {
        let range_min = extent.minimum >> self.chunk_shape_log2;
        let range_max = extent.max() >> self.chunk_shape_log2;
        let shape_log2 = self.chunk_shape_log2;

        ExtentN::from_min_and_max(range_min, range_max)
            .iter_points()
            .map(move |p| p << shape_log2)
    }

    /// The extent spanned by the chunk at `min`. Only applies to the LOD in which the chunk resides.
    #[inline]
    pub fn extent_for_chunk_with_min(&self, min: PointN<N>) -> ExtentN<N> {
        ExtentN::from_min_and_shape(min, self.chunk_shape)
    }

    /// The LOD0 extent covered by the chunk at `key` in a lower `lod`.
    #[inline]
    pub fn chunk_extent_at_lower_lod(&self, key: ChunkKey<N>, lod: u8) -> ExtentN<N> {
        debug_assert!(key.lod >= lod);
        self.extent_for_chunk_with_min(key.minimum) << (key.lod - lod) as i32
    }

    #[inline]
    pub fn ancestor_chunk_min(&self, p: PointN<N>, levels: i32) -> PointN<N> {
        debug_assert!(levels >= 0);
        (p & (self.chunk_shape_mask << levels)) >> levels
    }

    /// Given a chunk at `key`, returns the `ChunkKey` of the ancestor chunk at `ancestor_lod`.
    #[inline]
    pub fn ancestor_chunk_key(&self, key: ChunkKey<N>, ancestor_lod: u8) -> ChunkKey<N> {
        let levels = ancestor_lod as i32 - key.lod as i32;
        ChunkKey::new(ancestor_lod, self.ancestor_chunk_min(key.minimum, levels))
    }

    /// Given a chunk at `key`, returns the `ChunkKey` of the parent chunk.
    #[inline]
    pub fn parent_chunk_key(&self, key: ChunkKey<N>) -> ChunkKey<N> {
        self.ancestor_chunk_key(key, key.lod + 1)
    }

    /// Given an `extent`, returns an extent `levels` up that overlaps all ancestors of chunks covered by `extent`.
    #[inline]
    pub fn covering_ancestor_extent(&self, extent: ExtentN<N>, levels: i32) -> ExtentN<N> {
        ExtentN::from_min_and_max(
            self.ancestor_chunk_min(extent.minimum, levels),
            self.ancestor_chunk_min(extent.max(), levels),
        )
    }

    /// Given the chunk at `key`, returns the `ChunkKey` of the child chunk with `corner_index`.
    #[inline]
    pub fn child_chunk_key(&self, key: ChunkKey<N>, corner_index: u8) -> ChunkKey<N> {
        debug_assert!(key.lod > 0);
        let child_min =
            (key.minimum << 1) + (PointN::corner_offset(corner_index) << self.chunk_shape_log2);
        ChunkKey::new(key.lod - 1, child_min)
    }

    /// Given a chunk at `chunk_min`, returns the corner index relative to its parent.
    #[inline]
    pub fn corner_index(&self, chunk_min: PointN<N>) -> u8 {
        let double_mask = self.chunk_shape_mask << 1;
        let double_parent_min = chunk_min & double_mask;
        let offset = (chunk_min - double_parent_min) >> self.chunk_shape_log2;
        offset.as_corner_index()
    }

    /// When downsampling a chunk at level `L`, the samples are used at the returned destination within level `L + 1`.
    #[inline]
    pub(crate) fn downsample_destination(
        &self,
        src_chunk_key: ChunkKey<N>,
    ) -> DownsampleDestination<N> {
        let double_mask = self.chunk_shape_mask << 1;
        let double_parent_min = src_chunk_key.minimum & double_mask;
        let dst_chunk_min = double_parent_min >> 1;
        let dst_offset = Local((src_chunk_key.minimum - double_parent_min) >> 1);
        DownsampleDestination {
            chunk_key: ChunkKey::new(src_chunk_key.lod + 1, dst_chunk_min),
            offset: dst_offset,
        }
    }

    /// Call `visitor` on all keys in the Moore neighborhood of `key`.
    #[inline]
    pub fn visit_neighbor_keys(
        &self,
        key: ChunkKey<N>,
        mut visitor: impl FnMut(ChunkKey<N>) -> bool,
    ) -> bool {
        for offset in PointN::moore_offsets().into_iter() {
            let neighbor_min = key.minimum + (offset << self.chunk_shape_log2);
            let neighbor_key = ChunkKey::new(key.lod, neighbor_min);
            if !visitor(neighbor_key) {
                return false;
            }
        }

        true
    }

    /// Call `visitor` on all children keys of `parent_key`.
    #[inline]
    pub fn visit_child_keys(
        &self,
        parent_key: ChunkKey<N>,
        mut visitor: impl FnMut(ChunkKey<N>, u8),
    ) {
        for child_i in 0..PointN::NUM_CORNERS {
            let child_key = self.child_chunk_key(parent_key, child_i);
            visitor(child_key, child_i);
        }
    }
}

/// The key for a chunk at a particular level of detail.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChunkKey<N> {
    /// The minimum point of the chunk.
    pub minimum: PointN<N>,
    /// The level of detail. From highest resolution at `0` to lowest resolution at `root_lod`.
    pub lod: u8,
}

// A few of these traits could be derived. But it seems that derive will not help the compiler infer trait bounds as well.

impl<N> Clone for ChunkKey<N>
where
    PointN<N>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            minimum: self.minimum.clone(),
            lod: self.lod,
        }
    }
}
impl<N> Copy for ChunkKey<N> where PointN<N>: Copy {}

impl<N> PartialEq for ChunkKey<N>
where
    PointN<N>: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.minimum == other.minimum && self.lod == other.lod
    }
}

impl<N> Eq for ChunkKey<N> where PointN<N>: PartialEq {}

impl<N> std::hash::Hash for ChunkKey<N>
where
    PointN<N>: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.minimum.hash(state);
        state.write_u8(self.lod);
    }
}

/// A 2-dimensional `ChunkKey`.
pub type ChunkKey2 = ChunkKey<[i32; 2]>;
/// A 3-dimensional `ChunkKey`.
pub type ChunkKey3 = ChunkKey<[i32; 3]>;

impl<N> ChunkKey<N> {
    pub fn new(lod: u8, chunk_minimum: PointN<N>) -> Self {
        Self {
            lod,
            minimum: chunk_minimum,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DownsampleDestination<N> {
    pub chunk_key: ChunkKey<N>,
    pub offset: Local<N>,
}

/// A newtype wrapper for `PointN` or `ExtentN` where each point represents exactly one chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChunkUnits<T>(pub T);

impl<N> ChunkUnits<PointN<N>>
where
    PointN<N>: IntegerPoint,
{
    pub fn chunk_min(&self, chunk_shape: PointN<N>) -> PointN<N> {
        chunk_shape * self.0
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

    #[test]
    fn parent_chunk_key() {
        let indexer = ChunkIndexer::new(Point3i::fill(4));
        let lod0_p = Point3i::fill(15);
        let lod0_min = indexer.min_of_chunk_containing_point(lod0_p);
        let lod0_key = ChunkKey::new(0, lod0_min);
        assert_eq!(lod0_key.minimum, Point3i::fill(12));
        let lod1_key = indexer.parent_chunk_key(lod0_key);
        assert_eq!(lod1_key.minimum, Point3i::fill(4));
        let lod2_key = indexer.parent_chunk_key(lod1_key);
        assert_eq!(lod2_key.minimum, Point3i::fill(0));

        assert_eq!(lod2_key, indexer.ancestor_chunk_key(lod0_key, 2))
    }

    #[test]
    fn child_chunk_key() {
        let indexer = ChunkIndexer::new(Point3i::fill(4));
        let lod2_key = ChunkKey::new(2, Point3i::fill(0));
        let lod1_key = indexer.child_chunk_key(lod2_key, 0b111);
        assert_eq!(lod1_key.minimum, Point3i::fill(4));
        let lod0_key = indexer.child_chunk_key(lod1_key, 0b111);
        assert_eq!(lod0_key.minimum, Point3i::fill(12));
    }

    #[test]
    fn downsample_destination() {
        let chunk_shape = Point3i::fill(16);
        let indexer = ChunkIndexer::new(chunk_shape);

        let src_key = ChunkKey::new(0, chunk_shape);
        let dst = indexer.downsample_destination(src_key);
        assert_eq!(
            dst,
            DownsampleDestination {
                chunk_key: ChunkKey::new(1, Point3i::ZERO),
                offset: Local(chunk_shape / 2),
            }
        );

        let src_key = ChunkKey::new(0, 2 * chunk_shape);
        let dst = indexer.downsample_destination(src_key);
        assert_eq!(
            dst,
            DownsampleDestination {
                chunk_key: ChunkKey::new(1, chunk_shape),
                offset: Local(Point3i::ZERO),
            }
        );
    }

    #[test]
    fn corner_index_matches_children() {
        let chunk_shape = PointN([4, 8, 16]);
        let indexer = ChunkIndexer::new(chunk_shape);

        let parent_chunk_key = ChunkKey::new(1, PointN([-4, 0, 16]));
        for corner_index in 0..Point3i::NUM_CORNERS {
            assert_eq!(
                indexer.corner_index(
                    indexer
                        .child_chunk_key(parent_chunk_key, corner_index)
                        .minimum
                ),
                corner_index
            );
        }
    }
}
