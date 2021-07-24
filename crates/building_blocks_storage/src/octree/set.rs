//! The `OctreeSet` type is a memory-efficient set of points organized hierarchically. Often referred to as a "hashed octree."
//!
//! Any node in the tree can be uniquely represented as an `Octant`, however this representation is not actually stored.
//! Similarly, an `OctreeNode` can be used to represent any node during traversal of the tree, but neither is it used for
//! storage.
//!
//! Every node is either a branch node or a leaf node. Branch nodes are the nodes that actually occupy a `(LocationCode, u8)`
//! entry in the hash map, where the `u8` is a child bitmask. Leaf nodes do not take up any space, but they are implied by the
//! structure of the branch nodes. When a branch node has a leaf node as a child, then we say that leaf node is "fat". A fat
//! leaf node is a leaf node representing a full octant, i.e. all points in that octant are present in the set. The union of all
//! fat leaf octants is equal to the whole set. We also say that fat leaf nodes can have children for the sake of iteration, and
//! we call them "thin" leaf nodes.
//!
//! # Use Cases
//!
//! The `OctreeSet` has many uses.
//!
//! One possible use case is to construct one using `OctreeSet::from_array3`, then insert it into an `OctreeDbvt` in order to
//! perform spatial queries like raycasting.
//!
//! The `OctreeSet` is also used in the `OctreeChunkIndex`, where each point represents a single chunk. This representation is
//! useful for level of detail algorithms like clipmap traversal because inner nodes may correspond to downsampled chunks.
//!
//! # Traversal
//!
//! `OctreeSet` supports two modes of traversal. One is using the visitor pattern via `OctreeVisitor`. You can either visit just
//! the branches and leaves, or you can also visit full sub-octants of a leaf octant.
//!
//! ## Nested Traversal
//!
//! ```
//! # let some_condition = |_: &OctreeNode| true;
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::{octree::set::*, prelude::*};
//!
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(32));
//! let voxels = Array3x1::fill(extent, true); // boring example
//! let octree = OctreeSet::from_array3(&voxels, Extent3i::from_min_and_shape(Point3i::fill(8), Point3i::fill(16)));
//!
//! octree.visit_branches_and_fat_leaves_in_preorder(&mut |node: &OctreeNode| {
//!     if some_condition(node) {
//!         // Found a particular subtree, now narrow the search using a different algorithm.
//!         node.visit_all_octants_in_preorder(&octree, &mut |_node: &OctreeNode| {
//!             VisitStatus::Continue
//!         });
//!
//!         VisitStatus::ExitEarly
//!     } else {
//!         VisitStatus::Continue
//!     }
//! });
//! ```
//!
//! ## Manual Node Traversal
//!
//! The other form of traversal is "node-based," which is slightly less efficient and more manual but also more flexible. See
//! the `OctreeSet::root_node`, `OctreeSet::child_node`, and `OctreeNode` documentation for details.

use crate::dev_prelude::*;

use building_blocks_core::prelude::*;

use core::ops::Deref;
use serde::{Deserialize, Serialize};
use std::fmt::Formatter;

/// A sparse set of voxel coordinates (3D integer points). Supports spatial queries.
///
/// The octree is a cube shape and the edge lengths can only be a power of 2, at most 64.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OctreeSet {
    extent: Extent3i,
    power: u8,
    root_exists: bool,
    // Save memory by using 2-byte OctreeNode codes as hash map keys instead of 64-bit node pointers. The total memory usage can
    // be approximated as 4 bytes per node, assuming the hashbrown table has 1 byte of overhead per entry.
    nodes: SmallKeyHashMap<LocationCode, ChildBitMask>,
}

impl OctreeSet {
    /// Make an empty set in the universe (domain) of `extent`.
    pub fn new_empty(extent: Extent3i) -> Self {
        Self::new_without_nodes(extent, false)
    }

    /// Make a full set in the universe (domain) of `extent`.
    pub fn new_full(extent: Extent3i) -> Self {
        Self::new_without_nodes(extent, true)
    }

    fn new_without_nodes(extent: Extent3i, root_exists: bool) -> Self {
        let power = Self::check_extent(&extent);

        Self {
            power,
            root_exists,
            extent,
            nodes: SmallKeyHashMap::default(),
        }
    }

    fn check_extent(extent: &Extent3i) -> u8 {
        assert!(extent.shape.dimensions_are_powers_of_2());
        assert!(extent.shape.is_cube());
        let power = extent.shape.x().trailing_zeros() as u8;
        // Constrained by 16-bit OctreeNode code.
        assert!(power > 0 && power <= 6);

        power
    }

    // TODO: from_height_map

    /// Constructs an `OctreeSet` which contains all of the points in `extent` which are not empty (as defined by the `IsEmpty`
    /// trait). `extent` must be cube-shaped with edge length being a power of 2. For power `P` where edge length is `2^P`, we
    /// must have `0 < P <= 6`, because there is a maximum fixed depth of the octree.
    pub fn from_array3<A, T>(array: &A, extent: Extent3i) -> Self
    where
        A: IndexedArray<[i32; 3]> + GetUnchecked<Stride, Item = T>,
        T: Clone + IsEmpty,
    {
        assert!(
            extent.is_subset_of(array.extent()),
            "{:?} does not contain {:?}; would cause access out-of-bounds",
            array.extent(),
            extent
        );

        let power = Self::check_extent(&extent);
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

        let mut nodes = SmallKeyHashMap::default();
        let min_local = Local(extent.minimum - array.extent().minimum);
        let root_minimum = array.stride_from_local_point(min_local);
        let root_code = LocationCode::ROOT;
        let (root_exists, _full) = Self::partition_array(
            root_code,
            root_minimum,
            edge_length,
            &corner_strides,
            array,
            &mut nodes,
        );

        Self {
            extent,
            power,
            root_exists,
            nodes,
        }
    }

    fn partition_array<A, T>(
        code: LocationCode,
        minimum: Stride,
        edge_length: i32,
        corner_strides: &[Stride],
        array: &A,
        nodes: &mut SmallKeyHashMap<LocationCode, ChildBitMask>,
    ) -> (bool, bool)
    where
        A: IndexedArray<[i32; 3]> + GetUnchecked<Stride, Item = T>,
        T: Clone + IsEmpty,
    {
        // Base case where the octant is a single voxel. The `OctreeNode` is invalid and unnecessary in this case; we avoid using
        // it by returning early.
        if edge_length == 1 {
            let exists = unsafe { !array.get_unchecked(minimum).is_empty() };
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
        let extended_code = code.extend();
        for (octant, offset) in octant_corner_strides.iter().enumerate() {
            let octant_min = minimum + *offset;
            let octant_code = extended_code.with_lowest_octant(octant as u16);
            let (child_exists, child_full) = Self::partition_array(
                octant_code,
                octant_min,
                half_edge_length,
                &octant_corner_strides,
                array,
                nodes,
            );
            child_bitmask |= (child_exists as u8) << octant;
            all_children_full &= child_full;
        }

        let exists = child_bitmask != 0;

        if exists && !all_children_full {
            nodes.insert(code, child_bitmask);
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
    pub fn octant(&self) -> OctreeOctant {
        OctreeOctant(Octant::new_unchecked(
            self.extent.minimum,
            self.edge_length(),
        ))
    }

    /// The extent spanned by the octree.
    pub fn extent(&self) -> &Extent3i {
        &self.extent
    }

    /// Returns `true` iff the octree contains zero points.
    pub fn is_empty(&self) -> bool {
        !self.root_exists
    }

    /// Same as `visit_branches_and_fat_leaves_in_preorder`, but visit only the octants that overlap `extent`.
    pub fn visit_branches_and_fat_leaves_for_extent_in_preorder(
        &self,
        extent: &Extent3i,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self.visit_branches_and_fat_leaves_in_preorder(&mut |node: &OctreeNode| {
            Self::extent_visitor(extent, visitor, node)
        })
    }

    /// Same as `visit_branches_and_fat_leaves_in_postorder`, but visit only the octants that overlap `extent`.
    pub fn visit_branches_and_fat_leaves_for_extent_in_postorder(
        &self,
        extent: &Extent3i,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self.visit_branches_and_fat_leaves_in_postorder(
            &|node| Self::extent_predicate(extent, node),
            visitor,
        )
    }

    /// Same as `visit_branches_and_fat_leaves_in_preorder`, but descendants of fat leaves are also visited.
    pub fn visit_all_octants_in_preorder(&self, visitor: &mut impl OctreeVisitor) -> VisitStatus {
        if !self.root_exists {
            return VisitStatus::Continue;
        }

        self._visit_all_octants_in_preorder(LocationCode::ROOT, self.octant(), visitor)
    }

    /// Same as `visit_branches_and_fat_leaves_in_postorder`, but descendants of fat leaves are also visited.
    pub fn visit_all_octants_in_postorder(
        &self,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if !self.root_exists {
            return VisitStatus::Continue;
        }

        self._visit_all_octants_in_postorder(LocationCode::ROOT, self.octant(), predicate, visitor)
    }

    /// Same as `visit_all_octants_in_preorder`, but only for octants overlapping `extent`.
    pub fn visit_all_octants_for_extent_in_preorder(
        &self,
        extent: &Extent3i,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self.visit_all_octants_in_preorder(&mut |node: &OctreeNode| {
            Self::extent_visitor(extent, visitor, node)
        })
    }

    /// Same as `visit_all_octants_in_postorder`, but only for octants overlapping `extent`.
    pub fn visit_all_octants_for_extent_in_postorder(
        &self,
        extent: &Extent3i,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self.visit_all_octants_in_postorder(&|node| Self::extent_predicate(extent, node), visitor)
    }

    fn extent_visitor(
        extent: &Extent3i,
        visitor: &mut impl OctreeVisitor,
        node: &OctreeNode,
    ) -> VisitStatus {
        if Self::extent_predicate(extent, node) {
            return VisitStatus::Stop;
        }

        visitor.visit_octant(node)
    }

    fn extent_predicate(extent: &Extent3i, node: &OctreeNode) -> bool {
        !Extent3i::from(node.octant.0)
            .intersection(extent)
            .is_empty()
    }

    fn _visit_all_octants_in_preorder(
        &self,
        code: LocationCode,
        octant: OctreeOctant,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self._visit_branches_and_fat_leaves_in_preorder(code, octant, &mut |node: &OctreeNode| {
            if node.is_full() {
                node.octant.visit_self_and_descendants_in_preorder(visitor)
            } else {
                visitor.visit_octant(node)
            }
        })
    }

    fn _visit_all_octants_in_postorder(
        &self,
        code: LocationCode,
        octant: OctreeOctant,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        self._visit_branches_and_fat_leaves_in_postorder(
            code,
            octant,
            predicate,
            &mut |node: &OctreeNode| {
                if node.is_full() {
                    node.octant.visit_self_and_descendants_in_postorder(visitor)
                } else {
                    visitor.visit_octant(node)
                }
            },
        )
    }

    /// Visit every branch and fat leaf in the octree. This is a pre-order traversal.
    pub fn visit_branches_and_fat_leaves_in_preorder(
        &self,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if !self.root_exists {
            return VisitStatus::Continue;
        }

        self._visit_branches_and_fat_leaves_in_preorder(LocationCode::ROOT, self.octant(), visitor)
    }

    /// Visit every branch and fat leaf in the octree. This is a post-order traversal.
    pub fn visit_branches_and_fat_leaves_in_postorder(
        &self,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if !self.root_exists {
            return VisitStatus::Continue;
        }

        self._visit_branches_and_fat_leaves_in_postorder(
            LocationCode::ROOT,
            self.octant(),
            predicate,
            visitor,
        )
    }

    // TODO: should golf this repetitive code a bit

    fn _visit_branches_and_fat_leaves_in_preorder(
        &self,
        code: LocationCode,
        octant: OctreeOctant,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        // Precondition: OctreeNode exists.

        // Base case where the octant is a single leaf voxel.
        if octant.is_single_voxel() {
            return visitor.visit_octant(&OctreeNode::leaf(octant));
        }

        // Continue traversal of this branch.

        let child_bitmask = if let Some(&child_bitmask) = self.nodes.get(&code) {
            child_bitmask
        } else {
            // Since we know that OctreeNode exists, but it's not in the nodes map, this means that we can assume the entire
            // octant is full. This is an implicit leaf node.
            return visitor.visit_octant(&OctreeNode::leaf(octant));
        };

        // Definitely not at a leaf node.
        let status = visitor.visit_octant(&OctreeNode::branch(octant, code, child_bitmask));
        if status != VisitStatus::Continue {
            return status;
        }

        let extended_code = code.extend();
        for child_index in 0..8 {
            if (child_bitmask & (1 << child_index)) == 0 {
                // This child does not exist.
                continue;
            }

            let child_octant = octant.child(child_index);
            let octant_code = extended_code.with_lowest_octant(child_index as u16);
            if self._visit_branches_and_fat_leaves_in_preorder(octant_code, child_octant, visitor)
                == VisitStatus::ExitEarly
            {
                return VisitStatus::ExitEarly;
            }
        }

        // Continue with the rest of the tree.
        VisitStatus::Continue
    }

    fn _visit_branches_and_fat_leaves_in_postorder(
        &self,
        code: LocationCode,
        octant: OctreeOctant,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        // Precondition: OctreeNode exists.

        let mut handle_leaf = |octant: OctreeOctant| {
            let node = OctreeNode::leaf(octant);

            if predicate(&node) {
                visitor.visit_octant(&node)
            } else {
                VisitStatus::Continue
            }
        };

        // Base case where the octant is a single leaf voxel.
        if octant.is_single_voxel() {
            return handle_leaf(octant);
        }

        // Continue traversal of this branch.

        let child_bitmask = if let Some(&child_bitmask) = self.nodes.get(&code) {
            child_bitmask
        } else {
            // Since we know that OctreeNode exists, but it's not in the nodes map, this means that we can assume the entire
            // octant is full. This is an implicit leaf node.
            return handle_leaf(octant);
        };

        // Definitely not at a leaf node.
        let node = OctreeNode::branch(octant, code, child_bitmask);

        if !predicate(&node) {
            // Even though this branch doesn't satisfy the predicate, this is postorder, so we might still want to visit
            // ancestors.
            return VisitStatus::Continue;
        }

        let mut visit_ancestors = true;
        let extended_code = code.extend();
        for child_index in 0..8 {
            if (child_bitmask & (1 << child_index)) == 0 {
                // This child does not exist.
                continue;
            }

            let child_octant = octant.child(child_index);
            let octant_code = extended_code.with_lowest_octant(child_index as u16);

            let status = self._visit_branches_and_fat_leaves_in_postorder(
                octant_code,
                child_octant,
                predicate,
                visitor,
            );

            match status {
                VisitStatus::ExitEarly => return VisitStatus::ExitEarly,
                VisitStatus::Stop => {
                    visit_ancestors = false;
                }
                _ => (),
            }
        }

        if visit_ancestors {
            visitor.visit_octant(&node)
        } else {
            VisitStatus::ExitEarly
        }
    }

    /// The `OctreeNode` of the root, if it exists.
    #[inline]
    pub fn root_node(&self) -> Option<OctreeNode> {
        if self.root_exists {
            Some(OctreeNode {
                code: LocationCode::ROOT,
                octant: self.octant(),
                child_bitmask: self.nodes.get(&LocationCode::ROOT).cloned().unwrap_or(0),
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

        let child_octant = parent.octant.child(child_index);

        if child_octant.is_single_voxel() {
            // The child is a leaf, so we don't need to extend the OctreeNode or look for a child bitmask.
            return Some(OctreeNode {
                code: LocationCode::LEAF,
                octant: child_octant,
                child_bitmask: 0,
            });
        }

        let child_code = parent.code.extend().with_lowest_octant(child_index as u16);

        let (code, child_bitmask) = if let Some(bitmask) = self.nodes.get(&child_code) {
            (child_code, *bitmask)
        } else {
            (LocationCode::LEAF, 0)
        };

        Some(OctreeNode {
            code,
            octant: child_octant,
            child_bitmask,
        })
    }

    /// Add all points from `extent` to the set.
    pub fn add_extent(&mut self, add_extent: &Extent3i) {
        let (root_exists, _full) = self._add_extent(
            LocationCode::ROOT,
            self.octant(),
            self.root_exists,
            add_extent,
        );
        self.root_exists = root_exists;
    }

    /// Returns `(exists, is_full)` booleans.
    fn _add_extent(
        &mut self,
        code: LocationCode,
        octant: OctreeOctant,
        already_exists: bool,
        add_extent: &Extent3i,
    ) -> (bool, bool) {
        if octant.is_single_voxel() {
            let intersects = add_extent.contains(octant.minimum());
            return (intersects, intersects || already_exists);
        }

        let octant_extent = Extent3i::from(octant.0);
        let octant_intersection = add_extent.intersection(&octant_extent);

        if octant_extent == octant_intersection {
            // The octant is a subset of the extent being added, so we can make it an implicit leaf.
            if already_exists {
                self.remove_subtree(&code, octant.exponent());
            }
            return (true, true);
        }

        let (mut child_bitmask, already_had_bitmask) =
            if let Some(&child_bitmask) = self.nodes.get(&code) {
                // Mixed branch node.
                (child_bitmask, true)
            } else if already_exists {
                // Implicit leaf node.
                return (true, true);
            } else {
                // New node.
                (0, false)
            };

        if octant_intersection.is_empty() {
            // Nothing to do for this octant.
            return (already_exists, false);
        }

        let mut all_children_full = true;
        let extended_code = code.extend();
        for child_index in 0..8 {
            let child_code = extended_code.with_lowest_octant(child_index as u16);
            let child_octant = octant.child(child_index);
            let child_already_exists = child_bitmask & (1 << child_index) != 0;
            let (child_exists_after_add, child_full) =
                self._add_extent(child_code, child_octant, child_already_exists, add_extent);
            child_bitmask |= (child_exists_after_add as u8) << child_index;
            all_children_full &= child_full;
        }

        if child_bitmask != 0 && !all_children_full {
            self.nodes.insert(code, child_bitmask);
        } else if already_had_bitmask && all_children_full {
            self.remove_subtree(&code, octant.exponent());
        }

        (true, all_children_full)
    }

    /// Subtract all points from `extent` from the set.
    pub fn subtract_extent(&mut self, sub_extent: &Extent3i) {
        if self.root_exists {
            self.root_exists = self._subtract_extent(LocationCode::ROOT, self.octant(), sub_extent);
        }
    }

    /// Returns `true` iff this octant exists after subtraction.
    fn _subtract_extent(
        &mut self,
        code: LocationCode,
        octant: OctreeOctant,
        sub_extent: &Extent3i,
    ) -> bool {
        // Precondition: octant is not already empty.

        if octant.is_single_voxel() {
            return !sub_extent.contains(octant.minimum());
        }

        let octant_extent = Extent3i::from(octant.0);
        let octant_intersection = sub_extent.intersection(&octant_extent);

        if octant_extent == octant_intersection {
            // The octant is a subset of the extent being subtracted, so we can remove the entire subtree.
            self.remove_subtree(&code, octant.exponent());
            return false;
        }

        if octant_intersection.is_empty() {
            return true;
        }

        // At this point, we could only remove a proper subset of the children.

        let (child_bitmask_before, was_full) = if let Some(&child_bitmask) = self.nodes.get(&code) {
            // Mixed branch node.
            (child_bitmask, false)
        } else {
            // This is an implicit leaf node, so all children are full.
            (0xFF, true)
        };

        let mut child_bitmask_after = 0;
        let extended_code = code.extend();
        for child_index in 0..8 {
            let child_already_exists = child_bitmask_before & (1 << child_index) != 0;
            if !child_already_exists {
                // Don't need to remove anything from this child.
                continue;
            }

            let child_code = extended_code.with_lowest_octant(child_index as u16);
            let child_octant = octant.child(child_index);
            let child_exists_after_subtraction =
                self._subtract_extent(child_code, child_octant, sub_extent);
            child_bitmask_after |= (child_exists_after_subtraction as u8) << child_index;
        }

        if child_bitmask_after == 0 {
            if !was_full {
                self.nodes.remove(&code);
            }

            false
        } else {
            if was_full || child_bitmask_before != child_bitmask_after {
                self.nodes.insert(code, child_bitmask_after);
            }

            true
        }
    }

    fn remove_subtree(&mut self, code: &LocationCode, level: u8) {
        if let Some(child_bitmask) = self.nodes.remove(code) {
            if level == 1 {
                return;
            }

            let extended_code = code.extend();
            for child_index in 0..8 {
                if child_bitmask & (1 << child_index) != 0 {
                    let child_code = extended_code.with_lowest_octant(child_index);
                    self.remove_subtree(&child_code, level - 1);
                }
            }
        }
    }

    pub fn visit_all_points(&self, mut visitor: impl FnMut(Point3i)) {
        self.visit_all_octants_in_preorder(&mut |node: &OctreeNode| {
            if node.is_full() {
                for p in Extent3i::from(*node.octant()).iter_points() {
                    visitor(p);
                }
            }

            VisitStatus::Continue
        });
    }

    /// Get all of the points in this set collected into a `Vec`.
    pub fn collect_all_points(&self) -> Vec<Point3i> {
        let mut points = Vec::new();
        self.visit_all_points(|p| points.push(p));

        points
    }
}

/// Represents a single non-empty octant in the octree. Can be used for manual traversal by calling `OctreeSet::get_child`.
#[derive(Clone, Copy, Debug)]
pub struct OctreeNode {
    octant: OctreeOctant,
    code: LocationCode,
    child_bitmask: ChildBitMask,
}

impl OctreeNode {
    #[inline]
    pub fn is_full(&self) -> bool {
        self.code == LocationCode::LEAF
    }

    #[inline]
    pub fn octant(&self) -> &Octant {
        &self.octant
    }

    #[inline]
    pub fn child_bitmask(&self) -> ChildBitMask {
        self.child_bitmask
    }

    /// Similar to `OctreeSet::visit_branches_and_fat_leaves`, but only for the subtree at this `OctreeNode`.
    #[inline]
    pub fn visit_branches_and_fat_leaves_in_preorder(
        &self,
        octree: &OctreeSet,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if self.is_full() {
            visitor.visit_octant(self)
        } else {
            octree._visit_branches_and_fat_leaves_in_preorder(self.code, self.octant, visitor)
        }
    }

    /// Similar to `OctreeSet::visit_branches_and_fat_leaves`, but only for the subtree at this `OctreeNode`.
    #[inline]
    pub fn visit_branches_and_fat_leaves_in_postorder(
        &self,
        octree: &OctreeSet,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if self.is_full() {
            visitor.visit_octant(self)
        } else {
            octree._visit_branches_and_fat_leaves_in_postorder(
                self.code,
                self.octant,
                predicate,
                visitor,
            )
        }
    }

    /// Similar to `OctreeSet::visit_all_octants_in_preorder`, but only for the subtree at this `OctreeNode`.
    #[inline]
    pub fn visit_all_octants_in_preorder(
        &self,
        octree: &OctreeSet,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if self.is_full() {
            self.octant.visit_self_and_descendants_in_preorder(visitor)
        } else {
            octree._visit_all_octants_in_preorder(self.code, self.octant, visitor)
        }
    }

    /// Similar to `OctreeSet::visit_all_octants_in_postorder`, but only for the subtree at this `OctreeNode`.
    #[inline]
    pub fn visit_all_octants_in_postorder(
        &self,
        octree: &OctreeSet,
        predicate: &impl Fn(&OctreeNode) -> bool,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if self.is_full() {
            self.octant.visit_self_and_descendants_in_postorder(visitor)
        } else {
            octree._visit_all_octants_in_postorder(self.code, self.octant, predicate, visitor)
        }
    }

    fn leaf(octant: OctreeOctant) -> Self {
        Self {
            octant,
            code: LocationCode::LEAF,
            child_bitmask: FULL_CHILD_BIT_MASK,
        }
    }

    fn branch(octant: OctreeOctant, code: LocationCode, child_bitmask: ChildBitMask) -> Self {
        Self {
            octant,
            code,
            child_bitmask,
        }
    }
}

pub type ChildBitMask = u8;

/// Just to be consistent, even though a leaf doesn't have child tree nodes, it is always "full".
const FULL_CHILD_BIT_MASK: ChildBitMask = 0xFF;

/// Uniquely identifies a OctreeNode in a given octree.
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
#[derive(Clone, Copy, Deserialize, Hash, Eq, PartialEq, Serialize)]
struct LocationCode(u16);

impl std::fmt::Debug for LocationCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "LocationCode({:#018b})", self.0)
    }
}

impl LocationCode {
    const ROOT: Self = Self(1);
    // Leaves don't need to store child bitmasks, so we can give them a sentinel code.
    const LEAF: Self = Self(0);

    fn extend(self) -> Self {
        Self(self.0 << 3)
    }

    fn with_lowest_octant(self, octant: u16) -> Self {
        Self(self.0 | octant)
    }
}

/// A cube-shaped extent which is an octant at some level of an octree. As a leaf node, it represents a totally full set of
/// points.
#[derive(Clone, Copy, Debug)]
pub struct OctreeOctant(pub Octant);

impl Deref for OctreeOctant {
    type Target = Octant;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl OctreeOctant {
    /// Returns the child octant, where `child_index` specifies the child as a number in `[0..7]` of the binary format `0bZYX`.
    #[inline]
    pub fn child(&self, child_index: u8) -> Self {
        let half_edge_length = self.edge_length() >> 1;

        Self(Octant::new_unchecked(
            self.minimum() + half_edge_length * Point3i::CUBE_CORNER_OFFSETS[child_index as usize],
            half_edge_length,
        ))
    }

    /// Visit `self` and all octants descending from `self` (children and children's children). This is a pre-order traversal.
    pub fn visit_self_and_descendants_in_preorder(
        self,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        let status = visitor.visit_octant(&OctreeNode::leaf(self));

        if self.is_single_voxel() || status != VisitStatus::Continue {
            return status;
        }

        for child_index in 0..8 {
            match self
                .child(child_index)
                .visit_self_and_descendants_in_preorder(visitor)
            {
                VisitStatus::Continue => (),
                VisitStatus::ExitEarly => return VisitStatus::ExitEarly,
                VisitStatus::Stop => continue,
            }
        }

        VisitStatus::Continue
    }

    /// Visit `self` and all octants descending from `self` (children and children's children). This is a post-order traversal.
    pub fn visit_self_and_descendants_in_postorder(
        self,
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        if self.is_single_voxel() {
            return visitor.visit_octant(&OctreeNode::leaf(self));
        }

        for child_index in 0..8 {
            match self
                .child(child_index)
                .visit_self_and_descendants_in_postorder(visitor)
            {
                VisitStatus::Continue => (),
                VisitStatus::ExitEarly => return VisitStatus::ExitEarly,
                VisitStatus::Stop => continue,
            }
        }

        visitor.visit_octant(&OctreeNode::leaf(self))
    }
}

pub trait OctreeVisitor {
    /// Visit any octant that contains points in the octree.
    fn visit_octant(&mut self, node: &OctreeNode) -> VisitStatus;
}

impl<F> OctreeVisitor for F
where
    F: FnMut(&OctreeNode) -> VisitStatus,
{
    #[inline]
    fn visit_octant(&mut self, node: &OctreeNode) -> VisitStatus {
        (self)(node)
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
    fn add_extents() {
        let domain = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(32));

        // No overlap, but they touch.
        let mut test = UpdateExtentTest::new_empty(domain);
        test.assert_extent_added(Extent3i::from_min_and_max(
            PointN([5, 0, 0]),
            PointN([9, 5, 5]),
        ));
        test.assert_extent_added(Extent3i::from_min_and_max(
            PointN([10, 0, 0]),
            PointN([14, 5, 5]),
        ));

        // With overlap.
        let mut test = UpdateExtentTest::new_empty(domain);
        test.assert_extent_added(Extent3i::from_min_and_max(
            Point3i::fill(8),
            Point3i::fill(12),
        ));
        test.assert_extent_added(Extent3i::from_min_and_max(
            Point3i::fill(10),
            Point3i::fill(15),
        ));
    }

    #[test]
    fn subtract_extents() {
        let domain = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(32));

        // No overlap, but they touch.
        let mut test = UpdateExtentTest::new_full(domain);
        test.assert_extent_subtracted(Extent3i::from_min_and_max(
            PointN([5, 0, 0]),
            PointN([9, 5, 5]),
        ));
        test.assert_extent_subtracted(Extent3i::from_min_and_max(
            PointN([10, 0, 0]),
            PointN([14, 5, 5]),
        ));

        // With overlap.
        let mut test = UpdateExtentTest::new_full(domain);
        test.assert_extent_subtracted(Extent3i::from_min_and_max(
            Point3i::fill(8),
            Point3i::fill(12),
        ));
        test.assert_extent_subtracted(Extent3i::from_min_and_max(
            Point3i::fill(10),
            Point3i::fill(15),
        ));
    }

    struct UpdateExtentTest {
        domain: Extent3i,
        set: OctreeSet,
        expected_array_set: Array3x1<bool>,
    }

    impl UpdateExtentTest {
        fn new_empty(domain: Extent3i) -> Self {
            Self {
                domain,
                set: OctreeSet::new_empty(domain),
                expected_array_set: Array3x1::fill(domain, false),
            }
        }

        fn new_full(domain: Extent3i) -> Self {
            Self {
                domain,
                set: OctreeSet::new_full(domain),
                expected_array_set: Array3x1::fill(domain, true),
            }
        }

        fn assert_extent_added(&mut self, add_extent: Extent3i) {
            let Self {
                domain,
                set,
                expected_array_set,
            } = self;

            set.add_extent(&add_extent);
            expected_array_set.fill_extent(&add_extent, true);

            set.assert_all_nodes_reachable();

            let mut mirror_array_set = Array3x1::fill(*domain, false);
            Self::fill_bool_array(&set, &mut mirror_array_set);

            assert_eq!(set, &OctreeSet::from_array3(expected_array_set, *domain));
            assert_eq!(&mirror_array_set, expected_array_set);
        }

        fn assert_extent_subtracted(&mut self, sub_extent: Extent3i) {
            let Self {
                domain,
                set,
                expected_array_set,
            } = self;

            set.subtract_extent(&sub_extent);
            expected_array_set.fill_extent(&sub_extent, false);

            set.assert_all_nodes_reachable();

            let mut mirror_array_set = Array3x1::fill(*domain, false);
            Self::fill_bool_array(&set, &mut mirror_array_set);

            assert_eq!(set, &OctreeSet::from_array3(expected_array_set, *domain));
            assert_eq!(&mirror_array_set, expected_array_set);
        }

        fn fill_bool_array(set: &OctreeSet, array: &mut Array3x1<bool>) {
            set.visit_branches_and_fat_leaves_in_preorder(&mut |node: &OctreeNode| {
                if node.is_full() {
                    array.fill_extent(&Extent3i::from(*node.octant()), true);
                }

                VisitStatus::Continue
            });
        }
    }

    #[test]
    fn octants_occupied_iff_not_empty() {
        let voxels = random_voxels();
        let octree = OctreeSet::from_array3(&voxels, *voxels.extent());

        octree.assert_all_nodes_reachable();

        let mut non_empty_voxels = HashSet::new();

        voxels.for_each(voxels.extent(), |p: Point3i, v: Voxel| {
            if !v.is_empty() {
                non_empty_voxels.insert(p);
            }
        });

        let mut octant_voxels = HashSet::new();

        octree.visit_branches_and_fat_leaves_in_preorder(&mut |node: &OctreeNode| {
            if node.is_full() {
                for p in Extent3i::from(*node.octant()).iter_points() {
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
                if node.is_full() {
                    for p in Extent3i::from(*node.octant()).iter_points() {
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

    fn random_voxels() -> Array3x1<Voxel> {
        let mut rng = rand::thread_rng();
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(64));

        Array3x1::fill_with(extent, |_| Voxel(rng.gen()))
    }

    #[derive(Clone, Copy)]
    struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }

    impl OctreeSet {
        fn assert_all_nodes_reachable(&self) {
            let num_reachable_nodes = if self.root_exists {
                self.count_nodes(LocationCode::ROOT, self.power())
            } else {
                0
            };

            assert_eq!(self.nodes.len(), num_reachable_nodes);
        }

        fn count_nodes(&self, code: LocationCode, level: u8) -> usize {
            if level == 0 {
                return 0;
            }

            let child_bitmask = if let Some(child_bitmask) = self.nodes.get(&code) {
                child_bitmask
            } else {
                return 0;
            };

            let mut nodes_sum = 1;
            let extended_location = code.extend();
            for child_index in 0..8 {
                if (child_bitmask & (1 << child_index)) == 0 {
                    // This child does not exist.
                    continue;
                }

                let octant_location = extended_location.with_lowest_octant(child_index as u16);
                nodes_sum += self.count_nodes(octant_location, level - 1);
            }

            nodes_sum
        }
    }
}
