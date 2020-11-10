//! The `Octree` type is a memory-efficient set of points.
//!
//! The typical workflow for using an `Octree` is to construct it from an `Array3`, then insert it
//! into an `OctreeDBVT` in order to perform spatial queries like raycasting.

use building_blocks_core::prelude::*;
use building_blocks_storage::{access::GetUncheckedRelease, prelude::*, IsEmpty};

use fnv::FnvHashMap;

/// A sparse set of voxel coordinates (3D integer points). Supports spatial queries.
///
/// The octree is a cube shape and the edge lengths can only be a power of 2, at most 64. When an
/// entire octant is full, it will be stored in a collapsed representation, so the leaves of the
/// tree can be differently sized octants.
pub struct Octree {
    extent: Extent3i,
    root_level: u8,
    root_exists: bool,
    // Save memory by using 2-byte location codes as hash map keys instead of 64-bit node pointers.
    // The total memory usage can be approximated as 3 bytes per node, assuming a hashbrown table.
    nodes: FnvHashMap<LocationCode, ChildBitMask>,
}

impl Octree {
    // TODO: from_height_map

    /// Constructs an `Octree` which contains all of the points in `extent` which are not empty (as
    /// defined by the `IsEmpty` trait). `extent` must be cube-shaped with edge length being a power
    /// of 2. For exponent E where edge length is 2^E, we must have `0 < E <= 6`, because there is a
    /// maximum fixed depth of the octree.
    pub fn from_array3<A, T>(array: &A, extent: Extent3i) -> Self
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, T>,
        T: Clone + IsEmpty,
    {
        assert!(extent.shape.dimensions_are_powers_of_2());
        assert!(extent.shape.is_cube());
        let power = extent.shape.x().trailing_zeros();
        // Constrained by 16-bit location code.
        assert!(power > 0 && power <= 6);

        let root_level = (power - 1) as u8;
        let edge_len = 1 << power;

        // These are the corners of the root octant, in local coordinates.
        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| Local(p * edge_len))
            .collect();
        // Convert into strides for indexing efficiency.
        let mut corner_strides = [Stride(0); 8];
        array.strides_from_local_points(&corner_offsets, &mut corner_strides);

        let mut nodes = FnvHashMap::default();
        let min_local = Local(extent.minimum - array.extent().minimum);
        let root_minimum = array.stride_from_local_point(&min_local);
        let root_location = LocationCode(1);
        let (root_exists, _full) = Self::partition_array(
            root_location,
            root_minimum,
            edge_len,
            &corner_strides,
            array,
            &mut nodes,
        );

        Octree {
            root_level,
            root_exists,
            extent,
            nodes,
        }
    }

    fn partition_array<A, T>(
        location: LocationCode,
        minimum: Stride,
        edge_len: i32,
        corner_strides: &[Stride],
        array: &A,
        nodes: &mut FnvHashMap<LocationCode, ChildBitMask>,
    ) -> (bool, bool)
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, T>,
        T: Clone + IsEmpty,
    {
        // Base case where the octant is a single voxel.
        if edge_len == 1 {
            let exists = !array.get_unchecked_release(minimum).is_empty();
            return (exists, exists);
        }

        let mut octant_corner_strides = [Stride(0); 8];
        for (child_corner, parent_corner) in
            octant_corner_strides.iter_mut().zip(corner_strides.iter())
        {
            *child_corner = Stride(parent_corner.0 >> 1);
        }

        let half_edge_len = edge_len >> 1;
        let mut child_bitmask = 0;
        let mut all_children_full = true;
        let extended_location = location.extend();
        for (octant, offset) in octant_corner_strides.iter().enumerate() {
            let octant_min = minimum + *offset;
            let octant_location = extended_location.with_lowest_octant(octant as u16);
            let (child_exists, child_full) = Self::partition_array(
                octant_location,
                octant_min,
                half_edge_len,
                &octant_corner_strides,
                array,
                nodes,
            );
            child_bitmask |= (child_exists as u8) << octant;
            all_children_full = all_children_full && child_full;
        }

        let exists = child_bitmask != 0;

        if exists && !all_children_full {
            nodes.insert(location, child_bitmask);
        }

        (exists, all_children_full)
    }

    pub fn edge_length(&self) -> i32 {
        1 << (self.root_level + 1)
    }

    /// The entire octant spanned by the octree.
    pub fn octant(&self) -> Octant {
        Octant {
            minimum: self.extent.minimum,
            edge_length: self.edge_length(),
        }
    }

    /// The extent spanned by the octree.
    pub fn extent(&self) -> &Extent3i {
        &self.extent
    }

    /// Returns `true` iff the octree contains zero points.
    pub fn is_empty(&self) -> bool {
        !self.root_exists
    }

    /// Visit every non-empty octant of the octree.
    pub fn visit(&self, visitor: &mut impl OctreeVisitor) -> VisitStatus {
        if !self.root_exists {
            return VisitStatus::Continue;
        }

        let minimum = self.extent.minimum;
        let edge_len = self.edge_length();
        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| p * edge_len)
            .collect();

        self._visit(LocationCode(1), minimum, edge_len, &corner_offsets, visitor)
    }

    fn _visit(
        &self,
        location: LocationCode,
        minimum: Point3i,
        edge_length: i32,
        corner_offsets: &[Point3i],
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        // Precondition: location exists.

        let octant = Octant {
            minimum,
            edge_length,
        };

        // Base case where the octant is a single leaf voxel.
        if edge_length == 1 {
            return visitor.visit_octant(octant, true);
        }

        // Continue traversal of this branch.

        let child_bitmask = if let Some(child_bitmask) = self.nodes.get(&location) {
            child_bitmask
        } else {
            // Since we know that location exists, but it's not in the nodes map, this means that we
            // can assume the entire octant is full. This is an implicit leaf node.
            return visitor.visit_octant(octant, true);
        };

        // Definitely not at a leaf node.
        let status = visitor.visit_octant(octant, false);
        if status != VisitStatus::Continue {
            return status;
        }

        let mut octant_corner_offsets = [PointN([0; 3]); 8];
        for (child_corner, parent_corner) in
            octant_corner_offsets.iter_mut().zip(corner_offsets.iter())
        {
            *child_corner = parent_corner.scalar_right_shift(1);
        }

        let half_edge_length = edge_length >> 1;
        let extended_location = location.extend();
        for (octant, offset) in octant_corner_offsets.iter().enumerate() {
            if (child_bitmask & (1 << octant)) == 0 {
                // This child does not exist.
                continue;
            }

            let octant_min = minimum + *offset;
            let octant_location = extended_location.with_lowest_octant(octant as u16);
            if self._visit(
                octant_location,
                octant_min,
                half_edge_length,
                &octant_corner_offsets,
                visitor,
            ) == VisitStatus::ExitEarly
            {
                return VisitStatus::ExitEarly;
            }
        }

        // Continue with the rest of the tree.
        VisitStatus::Continue
    }
}

type ChildBitMask = u8;

/// Uniquely identifies a location in a given octree.
///
/// Supports an octree with at most 6 levels.
/// ```text
/// level N:
///   loc = 0b1
/// level N-1:
///   loc = 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111
/// level N-2:
///   loc = 0b1000000, ...
/// ...
/// level N-5:
///   loc = 0b1000000000000000, ...
/// ```
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct LocationCode(u16);

impl LocationCode {
    pub fn extend(self) -> Self {
        LocationCode(self.0 << 3)
    }

    pub fn with_lowest_octant(self, octant: u16) -> Self {
        LocationCode(self.0 | octant)
    }
}

/// A cube-shaped extent which is an octant at some level of an octree. As a leaf node, it
/// represents a totally full set of points.
#[derive(Clone, Copy)]
pub struct Octant {
    pub minimum: Point3i,
    pub edge_length: i32,
}

impl From<Octant> for Extent3i {
    fn from(octant: Octant) -> Self {
        Extent3i::from_min_and_shape(octant.minimum, PointN([octant.edge_length; 3]))
    }
}

pub trait OctreeVisitor {
    /// Visit any octant that contains points in the octree.
    fn visit_octant(&mut self, octant: Octant, is_leaf: bool) -> VisitStatus;
}

impl<F> OctreeVisitor for F
where
    F: FnMut(Octant, bool) -> VisitStatus,
{
    fn visit_octant(&mut self, octant: Octant, is_leaf: bool) -> VisitStatus {
        (self)(octant, is_leaf)
    }
}

#[derive(Eq, PartialEq)]
pub enum VisitStatus {
    /// Continue traversing this branch.
    Continue,
    /// Stop traversing this branch.
    Stop,
    /// Stop traversing the entire tree. No further nodes will be visited.
    ExitEarly,
}

#[cfg(feature = "ncollide")]
mod ncollide_support {
    use super::*;
    use crate::na_conversions::na_point3f_from_point3i;

    use ncollide3d::bounding_volume::AABB;

    impl Octant {
        pub fn aabb(&self) -> AABB<f32> {
            let aabb_min = na_point3f_from_point3i(self.minimum);
            let aabb_max = na_point3f_from_point3i(self.minimum + PointN([self.edge_length; 3]));

            AABB::new(aabb_min, aabb_max)
        }
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

    use rand::Rng;
    use std::collections::HashSet;

    #[test]
    fn octants_occupied_iff_not_empty() {
        let voxels = random_voxels();
        let octree = Octree::from_array3(&voxels, *voxels.extent());

        let mut non_empty_voxels = HashSet::new();

        voxels.for_each(voxels.extent(), |p: Point3i, v: Voxel| {
            if !v.is_empty() {
                non_empty_voxels.insert(p);
            }
        });

        let mut octant_voxels = HashSet::new();

        octree.visit(&mut |octant: Octant, is_leaf: bool| {
            if is_leaf {
                voxels.for_each(&Extent3i::from(octant), |p, _v| {
                    octant_voxels.insert(p);
                });
            }

            VisitStatus::Continue
        });

        assert_eq!(non_empty_voxels, octant_voxels);
    }

    fn random_voxels() -> Array3<Voxel> {
        let mut rng = rand::thread_rng();
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));

        Array3::fill_with(extent, |_| Voxel(rng.gen()))
    }

    #[derive(Clone)]
    struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }
}
