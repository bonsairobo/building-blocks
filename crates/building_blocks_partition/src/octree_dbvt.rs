//! The analog of `ncollide3d::DBVT` for voxel octrees.
//!
//! This structure works well in tandem with a `ChunkMap3`, where an `Octree` can be generated from
//! a chunk and subsequently placed into the `OctreeDBVT`.

use crate::octree::{Octant, Octree, OctreeVisitor, VisitStatus};

use core::hash::Hash;
use fnv::FnvHashMap;
use ncollide3d::{
    bounding_volume::AABB,
    partitioning::{self as nc_part, DBVTLeaf, DBVTLeafId, BVH, DBVT},
};

/// An ncollide `DBVT` containing `Octree`s. This turns the bounded `Octree` into an unbounded
/// acceleration structure. You may use whatever key type `K` to uniquely identify the octrees.
pub struct OctreeDBVT<K> {
    dbvt: DBVT<f32, Octree, AABB<f32>>,
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
    pub fn insert(&mut self, key: K, octree: Octree) -> Option<Octree> {
        let aabb = octree.octant().aabb();
        let new_leaf_id = self.dbvt.insert(DBVTLeaf::new(aabb, octree));

        self.leaf_ids
            .insert(key, new_leaf_id)
            .map(|old_leaf_id| self.dbvt.remove(old_leaf_id).data)
    }

    /// Visit every bounding volume (AABB) in the DBVT. This is a heterogeneous tree, meaning that
    /// not all nodes have the same representation. Upper nodes simply store a bounding volume
    /// (AABB), while octree nodes will provide both a bounding volume and an `Octant`, which is
    /// completely full for leaf nodes.
    pub fn visit(&self, visitor: &mut impl OctreeDBVTVisitor) {
        self.dbvt.visit(&mut DBVTVisitorImpl(visitor));
    }
}

struct DBVTVisitorImpl<'a, V>(&'a mut V);

impl<'a, V> nc_part::Visitor<Octree, AABB<f32>> for DBVTVisitorImpl<'a, V>
where
    V: OctreeDBVTVisitor,
{
    fn visit(&mut self, aabb: &AABB<f32>, octree: Option<&Octree>) -> nc_part::VisitStatus {
        let status = if let Some(octree) = octree {
            octree.visit(self.0)
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

impl<V> OctreeVisitor for V
where
    V: OctreeDBVTVisitor,
{
    fn visit_octant(&mut self, octant: Octant, is_leaf: bool) -> VisitStatus {
        let aabb = octant.aabb();

        self.visit(&aabb, Some(&octant), is_leaf)
    }
}

pub trait OctreeDBVTVisitor: OctreeVisitor {
    /// `octant` is only `Some` when traversing an `Octree`. Otherwise, you are traversing an
    /// upper-level internal node.
    fn visit(&mut self, aabb: &AABB<f32>, octant: Option<&Octant>, is_leaf: bool) -> VisitStatus;
}
