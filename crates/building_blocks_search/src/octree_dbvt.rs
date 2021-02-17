//! The analog of `ncollide3d::DBVT` for voxel octrees.
//!
//! This structure works well in tandem with a `ChunkMap3`, where an `Octree` can be generated from a chunk and subsequently
//! placed into the `OctreeDBVT`.

use building_blocks_core::prelude::*;
use building_blocks_storage::{Octant, OctreeNode, OctreeSet, OctreeVisitor, VisitStatus};

use core::hash::Hash;
use fnv::FnvHashMap;
use ncollide3d::{
    bounding_volume::AABB,
    partitioning::{self as nc_part, DBVTLeaf, DBVTLeafId, BVH, DBVT},
};

/// An ncollide `DBVT` containing `OctreeSet`s. This turns the bounded `Octree` into an unbounded acceleration structure. You
/// may use whatever key type `K` to uniquely identify the octrees.
pub struct OctreeDBVT<K> {
    dbvt: DBVT<f32, OctreeSet, AABB<f32>>,
    leaf_ids: FnvHashMap<K, DBVTLeafId>,
}

impl<K> Default for OctreeDBVT<K> {
    fn default() -> Self {
        Self {
            dbvt: DBVT::new(),
            leaf_ids: Default::default(),
        }
    }
}

impl<K> OctreeDBVT<K>
where
    K: Eq + Hash,
{
    /// Inserts the octree, replacing any old octree at `key` and returning it.
    pub fn insert(&mut self, key: K, octree: OctreeSet) -> Option<OctreeSet> {
        let aabb = octant_aabb(&octree.octant());
        let new_leaf_id = self.dbvt.insert(DBVTLeaf::new(aabb, octree));

        self.leaf_ids
            .insert(key, new_leaf_id)
            .map(|old_leaf_id| self.dbvt.remove(old_leaf_id).data)
    }

    /// Remove the octree at `key`.
    pub fn remove(&mut self, key: &K) -> Option<OctreeSet> {
        self.leaf_ids
            .remove(key)
            .map(|leaf_id| self.dbvt.remove(leaf_id).data)
    }

    /// Get a reference to the `OctreeSet` at `key`.
    pub fn get(&self, key: &K) -> Option<&OctreeSet> {
        self.leaf_ids
            .get(key)
            .and_then(|leaf_id| self.dbvt.get(*leaf_id).map(|leaf| &leaf.data))
    }

    /// Visit every bounding volume (AABB) in the DBVT. This is a heterogeneous tree, meaning that not all nodes have the same
    /// representation. Upper nodes simply store a bounding volume (AABB), while octree nodes will provide both a bounding
    /// volume and an `Octant`, which is completely full for leaf nodes.
    pub fn visit(&self, visitor: &mut impl OctreeDBVTVisitor) {
        self.dbvt.visit(&mut DBVTVisitorImpl(visitor));
    }
}

struct DBVTVisitorImpl<'a, V>(&'a mut V);

impl<'a, V> OctreeVisitor for DBVTVisitorImpl<'a, V>
where
    V: OctreeDBVTVisitor,
{
    fn visit_octant(&mut self, node: &OctreeNode) -> VisitStatus {
        let aabb = octant_aabb(&node.octant());

        self.0.visit(&aabb, Some(node.octant()), node.is_full())
    }
}

impl<'a, V> nc_part::Visitor<OctreeSet, AABB<f32>> for DBVTVisitorImpl<'a, V>
where
    V: OctreeDBVTVisitor,
{
    fn visit(&mut self, aabb: &AABB<f32>, octree: Option<&OctreeSet>) -> nc_part::VisitStatus {
        let status = if let Some(octree) = octree {
            octree.visit_branches_and_leaves_in_preorder(self)
        } else {
            self.0.visit(aabb, None, false)
        };

        match status {
            VisitStatus::Continue => nc_part::VisitStatus::Continue,
            VisitStatus::Stop => nc_part::VisitStatus::Stop,
            VisitStatus::ExitEarly => nc_part::VisitStatus::ExitEarly,
        }
    }
}

pub trait OctreeDBVTVisitor {
    /// `octant` is only `Some` when traversing an `Octree`. Otherwise, you are traversing an upper-level internal node.
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_full: bool) -> VisitStatus;
}

/// Returns the axis-aligned bounding box that bounds `octant`.
pub fn octant_aabb(octant: &Octant) -> AABB<f32> {
    let aabb_min = Point3f::from(octant.minimum()).into();
    let aabb_max = Point3f::from(octant.minimum() + Point3i::fill(octant.edge_length())).into();

    AABB::new(aabb_min, aabb_max)
}
