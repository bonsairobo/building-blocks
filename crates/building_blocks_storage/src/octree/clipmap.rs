use crate::{ChunkKey, ChunkKey3, ChunkUnits3, Octant, OctreeNode, OctreeSet, VisitStatus};

use building_blocks_core::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClipMapConfig3 {
    /// The number of levels of detail.
    num_lods: u8,
    /// The radius (in chunks) of a clipbox at any level of detail.
    clip_box_radius: i32,
    /// The shape of every chunk, regardless of LOD. Note that while a chunk at a higher LOD takes up more world space, it has
    /// the same shape as chunks at lower levels, because the voxel size also changes.
    ///
    /// **WARNING**: As of now, chunks must be cubes.
    chunk_shape: Point3i,
}

impl ClipMapConfig3 {
    pub fn new(num_lods: u8, clip_box_radius: u16, chunk_shape: Point3i) -> Self {
        assert!(clip_box_radius >= 2); // Radius 1 doesn't work for any more than a single LOD, so why are you using a clipmap?
        assert!(chunk_shape.dimensions_are_powers_of_2());

        Self {
            num_lods,
            clip_box_radius: clip_box_radius as i32,
            chunk_shape,
        }
    }

    pub fn chunk_edge_length_log2(&self) -> i32 {
        assert!(self.chunk_shape.is_cube());

        self.chunk_shape.x().trailing_zeros() as i32
    }
}

/// Traverse `octree` to find the `ChunkKey3`s that are "active" when the clipmap is centered at `lod0_center`. `active_rx`
/// is a callback that receives the chunk keys for active chunks.
pub fn active_clipmap_lod_chunks(
    config: &ClipMapConfig3,
    octree: &OctreeSet,
    lod0_center: ChunkUnits3,
    mut active_rx: impl FnMut(ChunkKey3),
) {
    let chunk_log2 = config.chunk_edge_length_log2();
    let centers = all_lod_centers(lod0_center.0, config.num_lods);

    let high_lod_boundary = config.clip_box_radius >> 1;

    octree.visit_all_octants_in_preorder(&mut |node: &OctreeNode| {
        let octant = node.octant();
        let lod = octant.power();
        if lod >= config.num_lods {
            return VisitStatus::Continue;
        }

        let offset_from_center = get_offset_from_lod_center(&octant, &centers);

        if lod == 0 || offset_from_center > high_lod_boundary {
            // println!(
            //     "lod = {:?} offset = {:?} octant = {:?}",
            //     lod, offset_from_center, octant
            // );
            // This octant can be rendered at this level of detail.
            active_rx(octant_chunk_key(chunk_log2, &octant));

            VisitStatus::Stop
        } else {
            // This octant should be rendered with more detail.
            VisitStatus::Continue
        }
    });
}

/// A notification that a chunk (at a particular level of detail) must be split or merged. This is usually the result of a
/// camera movement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodChunkUpdate<N> {
    Split(SplitChunk<N>),
    Merge(MergeChunks<N>),
}

/// A 3-dimensional `LodChunkUpdate`.
pub type LodChunkUpdate3 = LodChunkUpdate<[i32; 3]>;

/// Split `old_chunk` into many `new_chunks`. The number of new chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitChunk<N> {
    pub old_chunk: ChunkKey<N>,
    pub new_chunks: Vec<ChunkKey<N>>,
}

/// Merge many `old_chunks` into `new_chunk`. The number of old chunks depends on how many levels of detail the octant has
/// moved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeChunks<N> {
    pub old_chunks: Vec<ChunkKey<N>>,
    pub new_chunk: ChunkKey<N>,
}

/// A transient object used for running the `find_chunk_updates` method on multiple octrees.
pub struct ClipMapUpdate3 {
    chunk_log2: i32,
    num_lods: u8,
    low_lod_boundary: i32,
    high_lod_boundary: i32,
    old_centers: Vec<Point3i>,
    new_centers: Vec<Point3i>,
}

impl ClipMapUpdate3 {
    /// Prepare to run the `find_chunk_updates` method after the clipmap center has moved from `old_lod0_center` to
    /// `new_lod0_center`.
    pub fn new(
        config: &ClipMapConfig3,
        old_lod0_center: ChunkUnits3,
        new_lod0_center: ChunkUnits3,
    ) -> Self {
        Self {
            chunk_log2: config.chunk_shape.x().trailing_zeros() as i32,
            num_lods: config.num_lods,
            low_lod_boundary: config.clip_box_radius,
            high_lod_boundary: config.clip_box_radius >> 1,
            old_centers: all_lod_centers(old_lod0_center.0, config.num_lods),
            new_centers: all_lod_centers(new_lod0_center.0, config.num_lods),
        }
    }

    /// Traverse `octree` and find all chunks that need to be split or merged based on the movement of the center of the
    /// clipmap.
    pub fn find_chunk_updates(
        &self,
        octree: &OctreeSet,
        mut update_rx: impl FnMut(LodChunkUpdate3),
    ) {
        octree.visit_all_octants_in_preorder(&mut |node: &OctreeNode| {
            let octant = node.octant();

            let lod = octant.power();
            if lod >= self.num_lods || lod == 0 {
                return VisitStatus::Continue;
            }

            let old_offset_from_center = get_offset_from_lod_center(&octant, &self.old_centers);
            let offset_from_center = get_offset_from_lod_center(&octant, &self.new_centers);

            if old_offset_from_center > self.high_lod_boundary
                && offset_from_center <= self.high_lod_boundary
            {
                // Increase the detail for this octant.
                // Create the higher detail in descendant octants.
                let old_chunk = octant_chunk_key(self.chunk_log2, &octant);
                let new_chunks = find_merge_or_split_descendants(
                    self.chunk_log2,
                    octree,
                    node,
                    &self.new_centers,
                    self.high_lod_boundary,
                );
                update_rx(LodChunkUpdate::Split(SplitChunk {
                    old_chunk,
                    new_chunks,
                }));

                VisitStatus::Stop
            } else if offset_from_center > self.high_lod_boundary
                && old_offset_from_center <= self.high_lod_boundary
            {
                // Decrease the detail for this octant.
                // Delete the higher detail in descendant octants.
                let new_chunk = octant_chunk_key(self.chunk_log2, &octant);
                let old_chunks = find_merge_or_split_descendants(
                    self.chunk_log2,
                    octree,
                    node,
                    &self.old_centers,
                    self.high_lod_boundary,
                );
                update_rx(LodChunkUpdate::Merge(MergeChunks {
                    old_chunks,
                    new_chunk,
                }));

                VisitStatus::Stop
            } else if offset_from_center > self.low_lod_boundary
                && old_offset_from_center > self.low_lod_boundary
            {
                VisitStatus::Stop
            } else {
                VisitStatus::Continue
            }
        });
    }
}

fn all_lod_centers(lod0_center: Point3i, num_lods: u8) -> Vec<Point3i> {
    let mut centers = vec![lod0_center; num_lods as usize];
    for i in 1..num_lods as usize {
        centers[i] = centers[i - 1] >> 1;
    }

    centers
}

fn find_merge_or_split_descendants(
    chunk_log2: i32,
    octree: &OctreeSet,
    node: &OctreeNode,
    centers: &[Point3i],
    high_lod_boundary: i32,
) -> Vec<ChunkKey3> {
    let mut matching_chunks = Vec::with_capacity(8);
    node.visit_all_octants_in_preorder(octree, &mut |node: &OctreeNode| {
        let lod = node.octant().power();
        let old_offset_from_center = get_offset_from_lod_center(node.octant(), centers);
        if lod == 0 || old_offset_from_center > high_lod_boundary {
            matching_chunks.push(octant_chunk_key(chunk_log2, node.octant()));

            VisitStatus::Stop
        } else {
            VisitStatus::Continue
        }
    });

    matching_chunks
}

fn get_offset_from_lod_center(octant: &Octant, centers: &[Point3i]) -> i32 {
    let lod = octant.power();
    let lod_p = octant.minimum() >> lod as i32;
    let lod_center = centers[lod as usize];

    clipmap_coordinates(lod_p - lod_center)
        .abs()
        .max_component()
}

/// For calculating offsets from the clipmap center, we need to bias any nonnegative components to make voxel coordinates
/// symmetric about the center.
///
/// ```text
/// Voxel Coordinates
///
///   -3  -2  -1   0   1   2   3
/// <--|---|---|---|---|---|---|-->
///
/// Clipmap Coordinates
///
///     -3  -2  -1   1   2   3
/// <--|---|---|---|---|---|---|-->
/// ```
fn clipmap_coordinates(p: Point3i) -> Point3i {
    p.map_components_unary(|c| if c >= 0 { c + 1 } else { c })
}

fn octant_chunk_key(chunk_log2: i32, octant: &Octant) -> ChunkKey3 {
    let lod = octant.power();

    ChunkKey {
        lod,
        minimum: (octant.minimum() << chunk_log2) >> lod as i32,
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
    use crate::{ChunkUnits, SmallKeyHashSet};

    use super::*;

    use pretty_assertions::assert_eq;
    use std::iter::FromIterator;

    #[test]
    fn active_chunks_in_lod0_and_lod1() {
        let config = ClipMapConfig3::new(NUM_LODS, CLIP_BOX_RADIUS, CHUNK_SHAPE);
        let lod0_center = ChunkUnits(Point3i::ZERO);

        let domain = Extent3i::from_min_and_shape(Point3i::fill(-16), Point3i::fill(32));
        let mut octree = OctreeSet::new_empty(domain);
        let filled_extent = Extent3i::from_min_and_shape(Point3i::fill(-4), Point3i::fill(8));
        octree.add_extent(&filled_extent);

        let mut active_keys = SmallKeyHashSet::new();
        active_clipmap_lod_chunks(&config, &octree, lod0_center, |key| {
            active_keys.insert(key);
        });

        let mut lod1_set = OctreeSet::new_empty(domain);
        lod1_set.add_extent(&Extent3i::from_min_and_shape(
            Point3i::fill(-2),
            Point3i::fill(4),
        ));
        lod1_set.subtract_extent(&Extent3i::from_min_and_shape(
            Point3i::fill(-1),
            Point3i::fill(2),
        ));

        let expected_keys = SmallKeyHashSet::from_iter(
            Extent3i::from_min_and_shape(Point3i::fill(-2), Point3i::fill(4))
                .iter_points()
                .map(|p| ChunkKey {
                    minimum: p * CHUNK_SHAPE,
                    lod: 0,
                })
                .chain(lod1_set.collect_points().into_iter().map(|p| ChunkKey {
                    minimum: p * CHUNK_SHAPE,
                    lod: 1,
                })),
        );

        assert_eq!(active_keys, expected_keys);
    }

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);
    const NUM_LODS: u8 = 2;
    const CLIP_BOX_RADIUS: u16 = 2;
}
