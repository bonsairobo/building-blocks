//! The `OctreeSet` type is a memory-efficient set of points.
//!
//! The typical workflow for using an `Octree` is to construct it from an `Array3`, then insert it
//! into an `OctreeDBVT` in order to perform spatial queries like raycasting.
//!
//! `OctreeSet` supports two modes of traversal. One is using the visitor pattern, which is the most
//! efficient. The other is "node-based," which is less efficient and more manual but also more
//! flexible.

use crate::{access::GetUncheckedRelease, prelude::*, IsEmpty};

use building_blocks_core::prelude::*;

use fnv::FnvHashMap;

/// A sparse set of voxel coordinates (3D integer points). Supports spatial queries.
///
/// The octree is a cube shape and the edge lengths can only be a power of 2, at most 64. When an
/// entire octant is full, it will be stored in a collapsed representation, so the leaves of the
/// tree can be differently sized octants.
pub struct OctreeSet {
    extent: Extent3i,
    power: u8,
    root_exists: bool,
    // Save memory by using 2-byte location codes as hash map keys instead of 64-bit node pointers.
    // The total memory usage can be approximated as 4 bytes per node, assuming the hashbrown table
    // has 1 byte of overhead per entry.
    nodes: FnvHashMap<LocationCode, ChildBitMask>,
}

impl OctreeSet {
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
        let power = extent.shape.x().trailing_zeros() as u8;
        // Constrained by 16-bit location code.
        assert!(power > 0 && power <= 6);

        let edge_length = 1 << power;

        // These are the corners of the root octant, in local coordinates.
        let mut corner_offsets = [Local(Point3i::ZERO); 8];
        for (&p, dst) in Point3i::CUBE_CORNER_OFFSETS
            .iter()
            .zip(corner_offsets.iter_mut())
        {
            *dst = Local(edge_length * p);
        }
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
            edge_length,
            &corner_strides,
            array,
            &mut nodes,
        );

        Self {
            power,
            root_exists,
            extent,
            nodes,
        }
    }

    fn partition_array<A, T>(
        location: LocationCode,
        minimum: Stride,
        edge_length: i32,
        corner_strides: &[Stride],
        array: &A,
        nodes: &mut FnvHashMap<LocationCode, ChildBitMask>,
    ) -> (bool, bool)
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, T>,
        T: Clone + IsEmpty,
    {
        // Base case where the octant is a single voxel. The `location` is invalid and unnecessary
        // in this case; we avoid using it by returning early.
        if edge_length == 1 {
            let exists = !array.get_unchecked_release(minimum).is_empty();
            return (exists, exists);
        }

        let mut octant_corner_strides = [Stride(0); 8];
        for (child_corner, parent_corner) in
            octant_corner_strides.iter_mut().zip(corner_strides.iter())
        {
            *child_corner = Stride(parent_corner.0 >> 1);
        }

        let half_edge_length = edge_length >> 1;
        let mut child_bitmask = 0;
        let mut all_children_full = true;
        let extended_location = location.extend();
        for (octant, offset) in octant_corner_strides.iter().enumerate() {
            let octant_min = minimum + *offset;
            let octant_location = extended_location.with_lowest_octant(octant as u16);
            let (child_exists, child_full) = Self::partition_array(
                octant_location,
                octant_min,
                half_edge_length,
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

    /// The exponent P such that `self.edge_length() = 2 ^ P`.
    pub fn power(&self) -> u8 {
        self.power
    }

    /// The length of any edge of the root octant.
    pub fn edge_length(&self) -> i32 {
        1 << self.power
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

        self._visit(LocationCode(1), self.octant(), visitor)
    }

    /// Same as `visit`, but visit only the octants that overlap `extent`.
    pub fn visit_extent(&self, extent: &Extent3i, visitor: &mut impl OctreeVisitor) -> VisitStatus {
        self.visit(&mut |octant: Octant, is_leaf: bool| {
            if Extent3i::from(octant).intersection(extent).is_empty() {
                return VisitStatus::Stop;
            }

            visitor.visit_octant(octant, is_leaf)
        })
    }

    fn _visit(
        &self,
        location: LocationCode,
        octant: Octant,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        // Precondition: location exists.

        // Base case where the octant is a single leaf voxel.
        if octant.edge_length == 1 {
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

        let extended_location = location.extend();
        for child_index in 0..8 {
            if (child_bitmask & (1 << child_index)) == 0 {
                // This child does not exist.
                continue;
            }

            let child_octant = octant.child(child_index);
            let octant_location = extended_location.with_lowest_octant(child_index as u16);
            if self._visit(octant_location, child_octant, visitor) == VisitStatus::ExitEarly {
                return VisitStatus::ExitEarly;
            }
        }

        // Continue with the rest of the tree.
        VisitStatus::Continue
    }

    /// The `OctreeNode` of the root, if it exists.
    #[inline]
    pub fn root_node(&self) -> Option<OctreeNode> {
        if self.root_exists {
            Some(OctreeNode {
                location: LocationCode(1),
                octant: self.octant(),
                child_bitmask: self.nodes.get(&LocationCode(1)).cloned().unwrap_or(0),
                power: self.power,
            })
        } else {
            None
        }
    }

    /// Returns the child `OctreeNode` of `parent` at the given `child_index`, where `0 < child_octant < 8`.
    #[inline]
    pub fn get_child(&self, parent: &OctreeNode, child_index: u8) -> Option<OctreeNode> {
        debug_assert!(child_index < 8);

        if parent.child_bitmask & (1 << child_index) == 0 {
            return None;
        }

        let child_power = parent.power - 1;
        let child_octant = parent.octant.child(child_index);

        if child_power == 0 {
            // The child is a leaf, so we don't need to extend the location or look for a child
            // bitmask.
            return Some(OctreeNode {
                location: LocationCode::LEAF,
                octant: child_octant,
                child_bitmask: 0,
                power: child_power,
            });
        }

        let child_location = parent
            .location
            .extend()
            .with_lowest_octant(child_index as u16);

        let (location, child_bitmask) = if let Some(bitmask) = self.nodes.get(&child_location) {
            (child_location, *bitmask)
        } else {
            (LocationCode::LEAF, 0)
        };

        Some(OctreeNode {
            location,
            octant: child_octant,
            child_bitmask,
            power: child_power,
        })
    }
}

/// Represents a single non-empty octant in the octree. Used for manual traversal by calling
/// `OctreeSet::get_child`.
#[derive(Clone, Copy, Debug)]
pub struct OctreeNode {
    location: LocationCode,
    octant: Octant,
    child_bitmask: ChildBitMask,
    power: u8,
}

impl OctreeNode {
    /// A leaf node is one whose octant is entirely full.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.location == LocationCode::LEAF
    }

    #[inline]
    pub fn octant(&self) -> Octant {
        self.octant
    }

    /// The power of a node is directly related to the edge length of octants at that level of the
    /// tree. So the power of the root node is `P` where `edge_length = 2 ^ P`. The power of any
    /// single-point leaf octant is 0, because `edge_length = 1 = 2 ^ 0`.
    #[inline]
    pub fn power(&self) -> u8 {
        self.power
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
    // Leaves don't need to store child bitmasks, so we can give them a sentinel code.
    const LEAF: Self = LocationCode(0);

    fn extend(self) -> Self {
        LocationCode(self.0 << 3)
    }

    fn with_lowest_octant(self, octant: u16) -> Self {
        LocationCode(self.0 | octant)
    }
}

/// A cube-shaped extent which is an octant at some level of an octree. As a leaf node, it
/// represents a totally full set of points.
#[derive(Clone, Copy, Debug)]
pub struct Octant {
    minimum: Point3i,
    edge_length: i32,
}

impl Octant {
    #[inline]
    pub fn minimum(&self) -> Point3i {
        self.minimum
    }

    #[inline]
    pub fn edge_length(&self) -> i32 {
        self.edge_length
    }

    /// Returns the child octant, where `child_index` specifies the child as a number in `[0..7]` of the binary format `0bZYX`.
    #[inline]
    pub fn child(&self, child_index: u8) -> Self {
        let half_edge_length = self.edge_length >> 1;

        Self {
            minimum: self.minimum
                + half_edge_length * Point3i::CUBE_CORNER_OFFSETS[child_index as usize],
            edge_length: half_edge_length,
        }
    }

    /// Visit `self` and all octants descending from `self` (children and children's children).
    pub fn visit_self_and_descendants(self, visitor: &mut impl OctantVisitor) -> VisitStatus {
        let status = visitor.visit_octant(self);

        if self.edge_length == 1 || status != VisitStatus::Continue {
            return status;
        }

        for child_index in 0..8 {
            match self.child(child_index).visit_self_and_descendants(visitor) {
                VisitStatus::Continue => (),
                VisitStatus::ExitEarly => return VisitStatus::ExitEarly,
                VisitStatus::Stop => continue,
            }
        }

        VisitStatus::Continue
    }
}

impl From<Octant> for Extent3i {
    #[inline]
    fn from(octant: Octant) -> Self {
        Extent3i::from_min_and_shape(octant.minimum, PointN([octant.edge_length; 3]))
    }
}

pub trait OctantVisitor {
    fn visit_octant(&mut self, octant: Octant) -> VisitStatus;
}

impl<F> OctantVisitor for F
where
    F: FnMut(Octant) -> VisitStatus,
{
    #[inline]
    fn visit_octant(&mut self, octant: Octant) -> VisitStatus {
        (self)(octant)
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
    #[inline]
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
        let octree = OctreeSet::from_array3(&voxels, *voxels.extent());

        let mut non_empty_voxels = HashSet::new();

        voxels.for_each(voxels.extent(), |p: Point3i, v: Voxel| {
            if !v.is_empty() {
                non_empty_voxels.insert(p);
            }
        });

        let mut octant_voxels = HashSet::new();

        octree.visit(&mut |octant: Octant, is_leaf: bool| {
            if is_leaf {
                for p in Extent3i::from(octant).iter_points() {
                    octant_voxels.insert(p);
                }
            }

            VisitStatus::Continue
        });

        assert_eq!(non_empty_voxels, octant_voxels);

        // Now do the same test with a manual node traversal.
        let mut octant_voxels = HashSet::new();

        let mut queue = vec![octree.root_node()];
        while !queue.is_empty() {
            if let Some(node) = queue.pop().unwrap() {
                if node.is_leaf() {
                    for p in Extent3i::from(node.octant()).iter_points() {
                        octant_voxels.insert(p);
                    }
                } else {
                    for octant in 0..8 {
                        queue.push(octree.get_child(&node, octant));
                    }
                }
            }
        }

        assert_eq!(non_empty_voxels, octant_voxels);
    }

    fn random_voxels() -> Array3<Voxel> {
        let mut rng = rand::thread_rng();
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([64; 3]));

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
