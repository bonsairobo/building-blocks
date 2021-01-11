use crate::{Array, GetUncheckedRelease, Local, Octant, SignedDistance, Stride};

use building_blocks_core::prelude::*;

use serde::{Deserialize, Serialize};

pub fn padded_adf_chunk_extent(extent: &Extent3i) -> Extent3i {
    extent.add_to_shape(PointN([1; 3]))
}

#[derive(Clone, Debug, Eq, Deserialize, PartialEq, Serialize)]
pub struct Adf {
    root_pointer: NodePointer,
    branches: Vec<BranchNode>,
    leaves: Vec<LeafNode>,
    extent: Extent3i,
    power: u8,
}

#[derive(Clone, Copy, Debug, Eq, Deserialize, PartialEq, Serialize)]
struct BranchNode {
    child_pointers: [u16; 8], // TODO: this is potentially a lot of wasted memory, maybe location codes are useful here
    distances: [i8; 8],
    child_mask: u8,
    leaf_mask: u8,
}

impl BranchNode {
    fn child_pointer(&self, edge_length: i32, octant: u8) -> NodePointer {
        let octant_bit = 1 << octant;

        NodePointer {
            index: self.child_pointers[octant as usize],
            is_some: self.child_mask & octant_bit != 0,
            is_leaf: self.leaf_mask & octant_bit != 0,
            edge_length,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Deserialize, PartialEq, Serialize)]
struct LeafNode {
    // TODO: we could make leaves store a 3x3x3 cube to likely save some memory by reducing duplicates
    distances: [i8; 8],
}

impl Adf {
    #[inline]
    pub fn extent(&self) -> &Extent3i {
        &self.extent
    }

    #[inline]
    pub fn power(&self) -> u8 {
        self.power
    }

    /// The length of any edge of the root octant.
    #[inline]
    pub fn edge_length(&self) -> i32 {
        1 << self.power
    }

    #[inline]
    pub fn octant(&self) -> Octant {
        Octant::new(self.extent.minimum, self.edge_length())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root_pointer.is_null()
    }

    #[inline]
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    #[inline]
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    // ███████╗██████╗  ██████╗ ███╗   ███╗     █████╗ ██████╗ ██████╗  █████╗ ██╗   ██╗
    // ██╔════╝██╔══██╗██╔═══██╗████╗ ████║    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
    // █████╗  ██████╔╝██║   ██║██╔████╔██║    ███████║██████╔╝██████╔╝███████║ ╚████╔╝
    // ██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║    ██╔══██║██╔══██╗██╔══██╗██╔══██║  ╚██╔╝
    // ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║    ██║  ██║██║  ██║██║  ██║██║  ██║   ██║
    // ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝

    pub fn from_array3<A, D>(
        sdf: &A,
        extent: Extent3i,
        voxel_edge_length: f32,
        error_tolerance: f32,
    ) -> Self
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, D>,
        D: SignedDistance,
    {
        let iter_extent = extent.add_to_shape(PointN([-1; 3]));

        assert!(iter_extent.shape.dimensions_are_powers_of_2());
        assert!(iter_extent.shape.is_cube());
        let power = iter_extent.shape.x().trailing_zeros() as u8;
        let edge_length = 1 << power;

        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| Local(p * edge_length))
            .collect();
        let mut corner_strides = [Stride(0); 8];
        sdf.strides_from_local_points(&corner_offsets, &mut corner_strides);

        let encoder = DistanceEncoder::new(edge_length as f32 * voxel_edge_length);

        let mut branches = Vec::new();
        let mut leaves = Vec::new();
        let min_local = Local(iter_extent.minimum - sdf.extent().minimum);
        let root_minimum = sdf.stride_from_local_point(&min_local);
        let (root_pointer, _tmp_node) = Self::partition_array(
            sdf,
            error_tolerance,
            encoder,
            0,
            root_minimum,
            edge_length,
            &corner_strides,
            &mut branches,
            &mut leaves,
        );

        Self {
            root_pointer,
            branches,
            leaves,
            extent,
            power,
        }
    }

    fn partition_array<A, D>(
        sdf: &A,
        error_tolerance: f32,
        encoder: DistanceEncoder,
        depth: u32,
        minimum: Stride,
        edge_length: i32,
        corner_strides: &[Stride],
        branches: &mut Vec<BranchNode>,
        leaves: &mut Vec<LeafNode>,
    ) -> (NodePointer, TemporaryNode<D>)
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, D>,
        D: SignedDistance,
    {
        // Base case where the octant is a single voxel.
        if edge_length == 1 {
            return (
                NodePointer::NULL,
                TemporaryNode::from_array(sdf, minimum, corner_strides),
            );
        }

        let mut next_corner_strides = [Stride(0); 8];
        for (child_stride, parent_stride) in
            next_corner_strides.iter_mut().zip(corner_strides.iter())
        {
            *child_stride = Stride(parent_stride.0 >> 1);
        }
        let next_edge_length = edge_length >> 1;

        let mut child_nodes = [TemporaryNode::default(); 8];
        let mut child_pointers = [std::u16::MAX; 8];
        let mut child_mask = 0;
        for ((octant, offset), (pointer, node)) in next_corner_strides
            .iter()
            .enumerate()
            .zip(child_pointers.iter_mut().zip(child_nodes.iter_mut()))
        {
            let octant_min = minimum + *offset;
            let (p, n) = Self::partition_array(
                sdf,
                error_tolerance,
                encoder.half_size(),
                depth + 1,
                octant_min,
                next_edge_length,
                &next_corner_strides,
                branches,
                leaves,
            );
            child_mask |= (p.is_some as u8) << octant;
            *pointer = p.index;
            *node = n;
        }

        let parent_node = TemporaryNode::parent(&child_nodes);

        if child_mask != 0 {
            return Self::create_leaf_nodes_and_parent_branch(
                encoder,
                edge_length,
                parent_node,
                child_mask,
                child_pointers,
                child_nodes,
                branches,
                leaves,
            );
        }

        if !parent_node.contains_isosurface {
            return (NodePointer::NULL, parent_node);
        }

        if depth > 4 {
            let trilinear_grid = copy_octants_to_trilinear_grid(&child_nodes);
            if trilinear_approximates_well(&trilinear_grid, error_tolerance) {
                return (NodePointer::NULL, parent_node);
            }
        }

        Self::create_leaf_nodes_and_parent_branch(
            encoder,
            edge_length,
            parent_node,
            child_mask,
            child_pointers,
            child_nodes,
            branches,
            leaves,
        )
    }

    fn create_leaf_nodes_and_parent_branch<D>(
        encoder: DistanceEncoder,
        edge_length: i32,
        parent_node: TemporaryNode<D>,
        mut child_mask: u8,
        mut child_pointers: [u16; 8],
        child_nodes: [TemporaryNode<D>; 8],
        branches: &mut Vec<BranchNode>,
        leaves: &mut Vec<LeafNode>,
    ) -> (NodePointer, TemporaryNode<D>)
    where
        D: SignedDistance,
    {
        let mut leaf_mask = 0;

        for (octant, child_node) in child_nodes.iter().enumerate() {
            let octant_bit = 1 << octant;
            if child_node.contains_isosurface && (child_mask & octant_bit) == 0 {
                child_pointers[octant] = Self::push_node(
                    LeafNode {
                        distances: encoder.encode_distances(&child_node.distances),
                    },
                    leaves,
                );
                child_mask |= octant_bit;
                leaf_mask |= octant_bit;
            }
        }

        (
            NodePointer {
                index: Self::push_node(
                    BranchNode {
                        distances: encoder.encode_distances(&parent_node.distances),
                        child_pointers,
                        child_mask,
                        leaf_mask,
                    },
                    branches,
                ),
                is_some: true,
                is_leaf: false,
                edge_length,
            },
            parent_node,
        )
    }

    fn push_node<T>(node: T, nodes: &mut Vec<T>) -> u16 {
        let index = nodes.len();
        debug_assert!(index < std::u16::MAX as usize);
        nodes.push(node);

        index as u16
    }

    // ██╗   ██╗██╗███████╗██╗████████╗    ██╗     ███████╗ █████╗ ██╗   ██╗███████╗███████╗
    // ██║   ██║██║██╔════╝██║╚══██╔══╝    ██║     ██╔════╝██╔══██╗██║   ██║██╔════╝██╔════╝
    // ██║   ██║██║███████╗██║   ██║       ██║     █████╗  ███████║██║   ██║█████╗  ███████╗
    // ╚██╗ ██╔╝██║╚════██║██║   ██║       ██║     ██╔══╝  ██╔══██║╚██╗ ██╔╝██╔══╝  ╚════██║
    //  ╚████╔╝ ██║███████║██║   ██║       ███████╗███████╗██║  ██║ ╚████╔╝ ███████╗███████║
    //   ╚═══╝  ╚═╝╚══════╝╚═╝   ╚═╝       ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚══════╝

    /// Truncate the tree to `max_depth` and visit the leaves.
    pub fn visit_leaves(&self, max_depth: usize, visitor: &mut impl AdfOctantVisitor) {
        if self.root_pointer.is_null() {
            return;
        }

        let octant = self.octant();
        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| p * octant.edge_length())
            .collect();
        let decoder = DistanceDecoder::new(octant.edge_length() as f32);
        self.visit_leaves_in_octant(
            decoder,
            &corner_offsets,
            octant,
            self.root_pointer.index,
            max_depth,
            visitor,
        )
    }

    fn visit_leaves_in_octant(
        &self,
        decoder: DistanceDecoder,
        corner_offsets: &[Point3i],
        octant: Octant,
        index: u16,
        levels_remaining: usize,
        visitor: &mut impl AdfOctantVisitor,
    ) {
        let branch = self.branches[index as usize];

        if levels_remaining == 0 {
            let distances = decoder.decode_distances(&branch.distances);
            visitor.visit_octant(AdfVisitNodeId::branch(index), octant, &distances);
            return;
        }

        let half_edge_length = octant.edge_length() >> 1;
        let half_decoder = decoder.half_size();

        let mut octant_corner_offsets = [PointN([0; 3]); 8];
        for (child_corner, parent_corner) in
            octant_corner_offsets.iter_mut().zip(corner_offsets.iter())
        {
            *child_corner = parent_corner.scalar_right_shift(1);
        }

        for ((octant_index, &child_pointer), offset) in branch
            .child_pointers
            .iter()
            .enumerate()
            .zip(octant_corner_offsets.iter())
        {
            let octant_bit = 1 << octant_index;
            if branch.child_mask & octant_bit == 0 {
                continue;
            }

            let child_octant = Octant::new(octant.minimum() + *offset, half_edge_length);

            if branch.leaf_mask & octant_bit == 0 {
                self.visit_leaves_in_octant(
                    half_decoder,
                    &octant_corner_offsets,
                    child_octant,
                    child_pointer,
                    levels_remaining - 1,
                    visitor,
                );
            } else {
                self.visit_leaf(half_decoder, child_octant, child_pointer, visitor);
            }
        }
    }

    fn visit_leaf(
        &self,
        decoder: DistanceDecoder,
        octant: Octant,
        index: u16,
        visitor: &mut impl AdfOctantVisitor,
    ) {
        let leaf = self.leaves[index as usize];
        let distances = decoder.decode_distances(&leaf.distances);
        visitor.visit_octant(AdfVisitNodeId::leaf(index), octant, &distances);
    }

    // ██╗   ██╗██╗███████╗██╗████████╗    ███╗   ███╗██╗███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗███████╗
    // ██║   ██║██║██╔════╝██║╚══██╔══╝    ████╗ ████║██║████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝
    // ██║   ██║██║███████╗██║   ██║       ██╔████╔██║██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  ███████╗
    // ╚██╗ ██╔╝██║╚════██║██║   ██║       ██║╚██╔╝██║██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║
    //  ╚████╔╝ ██║███████║██║   ██║       ██║ ╚═╝ ██║██║██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗███████║
    //   ╚═══╝  ╚═╝╚══════╝╚═╝   ╚═╝       ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝    ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝

    // PERF: try changing all of these const u8s to usizes
    // PERF: try destructuring arrays with static size instead of indexing

    /// Only visits minimal edges that intersect the isosurface.
    /// The edge-adjacent nodes given to the visitor are guaranteed to be in Z order.
    pub fn visit_minimal_edges(&self, visitor: &mut impl AdfEdgeVisitor) {
        if self.root_pointer.is_some {
            self.visit_minimal_edges_in_octant(
                self.edge_length(),
                &self.branches[self.root_pointer.index as usize],
                visitor,
            )
        }
    }

    fn visit_minimal_edges_in_octant(
        &self,
        edge_length: i32,
        branch: &BranchNode,
        visitor: &mut impl AdfEdgeVisitor,
    ) {
        let half_edge_length = edge_length >> 1;

        for (octant, &child_pointer) in branch.child_pointers.iter().enumerate() {
            let octant_bit = 1 << octant;
            if branch.child_mask & octant_bit != 0 && branch.leaf_mask & octant_bit == 0 {
                self.visit_minimal_edges_in_octant(
                    half_edge_length,
                    &self.branches[child_pointer as usize],
                    visitor,
                );
            }
        }

        for d in 0..3 {
            for face in 0..4 {
                let o = FACE_ADJACENT_OCTANTS[d][face];
                let p = [
                    branch.child_pointer(half_edge_length, o[0]),
                    branch.child_pointer(half_edge_length, o[1]),
                ];
                self.visit_minimal_edges_in_face(d, p, visitor);
            }

            for edge in 0..2 {
                let o = EDGE_ADJACENT_OCTANTS[d][edge];
                let p = [
                    branch.child_pointer(half_edge_length, o[0]),
                    branch.child_pointer(half_edge_length, o[1]),
                    branch.child_pointer(half_edge_length, o[2]),
                    branch.child_pointer(half_edge_length, o[3]),
                ];
                self.visit_minimal_edges_in_edge(d, p, visitor);
            }
        }
    }

    fn visit_minimal_edges_in_face(
        &self,
        d: usize,            // direction
        p: [NodePointer; 2], // face-adjacent node pointers
        visitor: &mut impl AdfEdgeVisitor,
    ) {
        // PRECONDITION: `p` nodes are given in increasing order (- side of face to + side).

        if p[0].is_null() || p[1].is_null() || (p[0].is_leaf && p[1].is_leaf) {
            return;
        }

        // We have two non-null nodes and at least one branch.

        // In both cases, mirror the octants because they have different parents:
        //
        // +--+--+   +--+--+
        // |  | x|   |x |  |
        // +--+--+   +--+--+
        // |  | x|   |x |  |
        // +--+--+   +--+--+
        //    p0        p1

        for face in 0..4 {
            let o = FACE_ADJACENT_OCTANTS[d][face];

            // Mirroring is independent of which face we're looking at.
            let next_p = [
                self.child_pointer(&p[0], o[1]),
                self.child_pointer(&p[1], o[0]),
            ];

            self.visit_minimal_edges_in_face(d, next_p, visitor);
        }

        for edge in 0..4 {
            let o = FACE_TO_EDGE_ADJACENT_OCTANTS[d][edge];

            // Depending on the orientation of the Z curve relative to the parent octants...
            //
            //      +---+   +---+      +---+   +---+
            //      | x-|---|-x |      | x |   | x |
            //      |   | / |   |  vs  | | | \ | | |
            //      | x-|---|-x |      | x |   | x |
            //      +---+   +---+      +---+   +---+
            //       p0      p1         p0      p1
            //
            // ...we have to change the order in which we:
            //
            // 1. select from the parent nodes.
            let order = FACE_TO_EDGE_NODE_ORDERS[edge];
            // 2. mirror across the face.
            let mirror = FACE_TO_EDGE_MIRRORS[edge];

            let next_p = [
                self.child_pointer(&p[order[0]], o[mirror[0]]),
                self.child_pointer(&p[order[1]], o[mirror[1]]),
                self.child_pointer(&p[order[2]], o[mirror[2]]),
                self.child_pointer(&p[order[3]], o[mirror[3]]),
            ];

            // "Face direction" is not the same as "edge direction."
            let edge_d = FACE_TO_EDGE_DIRECTION[d][edge];

            self.visit_minimal_edges_in_edge(edge_d, next_p, visitor);
        }
    }

    fn visit_minimal_edges_in_edge(
        &self,
        d: usize,            // direction
        p: [NodePointer; 4], // edge-adjacent nodes
        visitor: &mut impl AdfEdgeVisitor,
    ) {
        // PRECONDITION: `p` nodes are given in Z order.
        //
        //     p0 p1    00 01
        //     p2 p3    10 11

        if p[0].is_null() || p[1].is_null() || p[2].is_null() || p[3].is_null() {
            return;
        }

        if p[0].is_leaf && p[1].is_leaf && p[2].is_leaf && p[3].is_leaf {
            self.visit_minimal_edge(d, p, visitor);
        } else {
            for edge in 0..2 {
                let o = EDGE_ADJACENT_OCTANTS[d][edge];

                // Swap diagonal octants because they have different parents:
                //
                // +--+--+  +--+--+
                // |  |  |  |  |  |
                // +--+--+  +--+--+
                // |  | x|  |x |  |
                // +--+--+  +--+--+
                //    p0       p1
                //
                // +--+--+  +--+--+
                // |  | x|  |x |  |
                // +--+--+  +--+--+
                // |  |  |  |  |  |
                // +--+--+  +--+--+
                //    p2       p3
                let next_p = [
                    self.child_pointer(&p[0], o[3]),
                    self.child_pointer(&p[1], o[2]),
                    self.child_pointer(&p[2], o[1]),
                    self.child_pointer(&p[3], o[0]),
                ];

                self.visit_minimal_edges_in_edge(d, next_p, visitor);
            }
        }
    }

    fn visit_minimal_edge(
        &self,
        d: usize,            // direction
        p: [NodePointer; 4], // edge-adjacent nodes
        visitor: &mut impl AdfEdgeVisitor,
    ) {
        // PRECONDITION: `p` nodes are given in Z order.
        //
        //     p0 p1    00 01
        //     p2 p3    10 11

        // Only consider the smallest edge shared by all nodes.
        let mut min_edge_length = std::i32::MAX;
        let mut min_index = std::usize::MAX;
        for (i, node) in p.iter().enumerate() {
            if node.edge_length < min_edge_length {
                min_edge_length = node.edge_length;
                min_index = i;
            }
        }

        // Select the edge at the opposite corner of the octant.
        let d_octants = EDGE_ADJACENT_OCTANTS[d];
        let opposite_corner = [3, 2, 1, 0][min_index];
        let c0 = d_octants[0][opposite_corner];
        let c1 = d_octants[1][opposite_corner];
        let distances = self.get_distances(&p[min_index]);
        let d0 = distances[c0 as usize];
        let d1 = distances[c1 as usize];

        let flip = match (d0 < 0, d1 < 0) {
            (true, false) => true,
            (false, true) => false,
            // No sign change on this edge.
            _ => return,
        };

        if flip {
            visitor.visit_minimal_edge([p[2].id(), p[3].id(), p[0].id(), p[1].id()]);
        } else {
            visitor.visit_minimal_edge([p[0].id(), p[1].id(), p[2].id(), p[3].id()]);
        }
    }

    fn child_pointer(&self, parent: &NodePointer, octant: u8) -> NodePointer {
        debug_assert!(parent.is_some);
        if parent.is_leaf {
            parent.clone()
        } else {
            self.branches[parent.index as usize].child_pointer(parent.edge_length >> 1, octant)
        }
    }

    fn get_distances(&self, p: &NodePointer) -> &[i8; 8] {
        if p.is_leaf {
            &self.leaves[p.index as usize].distances
        } else {
            &self.branches[p.index as usize].distances
        }
    }
}

// Pairs of octants (o0, o1) where o0 and o1 are face-adjacent. o0 and o1 have different parents.
const FACE_ADJACENT_OCTANTS: [[[u8; 2]; 4]; 3] = [
    [
        //  -X     +X
        [0b000, 0b001],
        [0b010, 0b011],
        [0b100, 0b101],
        [0b110, 0b111],
    ],
    [
        //  -Y     +Y
        [0b000, 0b010],
        [0b001, 0b011],
        [0b100, 0b110],
        [0b101, 0b111],
    ],
    [
        //  -Z     +Z
        [0b000, 0b100],
        [0b001, 0b101],
        [0b010, 0b110],
        [0b011, 0b111],
    ],
];

// Quartets of octants (o0, o1, o2, o3) where all of o0-o4 are edge-adjacent.
// Ordering of octants in a quartet is consistently Z order.
const EDGE_ADJACENT_OCTANTS_NEG_X: [u8; 4] = [0b000, 0b010, 0b100, 0b110];
const EDGE_ADJACENT_OCTANTS_POS_X: [u8; 4] = [0b001, 0b011, 0b101, 0b111];
const EDGE_ADJACENT_OCTANTS_NEG_Y: [u8; 4] = [0b000, 0b100, 0b001, 0b101];
const EDGE_ADJACENT_OCTANTS_POS_Y: [u8; 4] = [0b010, 0b110, 0b011, 0b111];
const EDGE_ADJACENT_OCTANTS_NEG_Z: [u8; 4] = [0b000, 0b001, 0b010, 0b011];
const EDGE_ADJACENT_OCTANTS_POS_Z: [u8; 4] = [0b100, 0b101, 0b110, 0b111];

const EDGE_ADJACENT_OCTANTS: [[[u8; 4]; 2]; 3] = [
    [EDGE_ADJACENT_OCTANTS_NEG_X, EDGE_ADJACENT_OCTANTS_POS_X],
    [EDGE_ADJACENT_OCTANTS_NEG_Y, EDGE_ADJACENT_OCTANTS_POS_Y],
    [EDGE_ADJACENT_OCTANTS_NEG_Z, EDGE_ADJACENT_OCTANTS_POS_Z],
];

const FACE_TO_EDGE_ADJACENT_OCTANTS: [[[u8; 4]; 4]; 3] = [
    [
        // X
        EDGE_ADJACENT_OCTANTS_NEG_Y,
        EDGE_ADJACENT_OCTANTS_POS_Y,
        EDGE_ADJACENT_OCTANTS_NEG_Z,
        EDGE_ADJACENT_OCTANTS_POS_Z,
    ],
    [
        // Y
        EDGE_ADJACENT_OCTANTS_NEG_Z,
        EDGE_ADJACENT_OCTANTS_POS_Z,
        EDGE_ADJACENT_OCTANTS_NEG_X,
        EDGE_ADJACENT_OCTANTS_POS_X,
    ],
    [
        // Z
        EDGE_ADJACENT_OCTANTS_NEG_X,
        EDGE_ADJACENT_OCTANTS_POS_X,
        EDGE_ADJACENT_OCTANTS_NEG_Y,
        EDGE_ADJACENT_OCTANTS_POS_Y,
    ],
];
const FACE_TO_EDGE_DIRECTION: [[usize; 4]; 3] = [[1, 1, 2, 2], [2, 2, 0, 0], [0, 0, 1, 1]];
const FACE_TO_EDGE_NODE_ORDERS: [[usize; 4]; 4] =
    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]];
const FACE_TO_EDGE_MIRRORS: [[usize; 4]; 4] =
    [[2, 3, 0, 1], [2, 3, 0, 1], [1, 0, 3, 2], [1, 0, 3, 2]];

// ███╗   ██╗ ██████╗ ██████╗ ███████╗    ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
// ████╗  ██║██╔═══██╗██╔══██╗██╔════╝    ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
// ██╔██╗ ██║██║   ██║██║  ██║█████╗      ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
// ██║╚██╗██║██║   ██║██║  ██║██╔══╝      ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
// ██║ ╚████║╚██████╔╝██████╔╝███████╗    ██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
// ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝

#[derive(Clone, Copy, Debug, Eq, Deserialize, PartialEq, Serialize)]
struct NodePointer {
    edge_length: i32,
    index: u16,
    is_some: bool,
    is_leaf: bool,
}

impl NodePointer {
    const NULL: Self = Self {
        edge_length: 0,
        index: std::u16::MAX,
        is_some: false,
        is_leaf: false,
    };

    fn is_null(&self) -> bool {
        !self.is_some
    }

    fn id(&self) -> AdfVisitNodeId {
        AdfVisitNodeId::new(self.index, self.is_leaf)
    }
}

#[derive(Clone, Copy)]
struct TemporaryNode<D> {
    distances: [D; 8],
    contains_isosurface: bool,
}

impl<D> Default for TemporaryNode<D>
where
    D: SignedDistance,
{
    fn default() -> Self {
        Self {
            distances: [D::ZERO; 8],
            contains_isosurface: false,
        }
    }
}

impl<D> TemporaryNode<D>
where
    D: SignedDistance,
{
    fn from_array<A>(sdf: &A, minimum: Stride, corner_strides: &[Stride]) -> Self
    where
        A: GetUncheckedRelease<Stride, D>,
    {
        let mut distances = [D::ZERO; 8];
        let mut num_negative = 0;
        for (&stride, distance) in corner_strides.iter().zip(distances.iter_mut()) {
            *distance = sdf.get_unchecked_release(minimum + stride);
            if distance.is_negative() {
                num_negative += 1;
            }
        }

        Self {
            distances,
            contains_isosurface: num_negative != 0 && num_negative != 8,
        }
    }

    fn parent(children: &[Self]) -> Self {
        let mut distances = [D::ZERO; 8];
        let mut contains_isosurface = false;
        for ((octant, child), distance) in children.iter().enumerate().zip(distances.iter_mut()) {
            *distance = child.distances[octant];
            contains_isosurface |= child.contains_isosurface;
        }

        Self {
            distances,
            contains_isosurface,
        }
    }
}

// ██████╗ ██╗███████╗████████╗ █████╗ ███╗   ██╗ ██████╗███████╗
// ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝██╔════╝
// ██║  ██║██║███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗
// ██║  ██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝
// ██████╔╝██║███████║   ██║   ██║  ██║██║ ╚████║╚██████╗███████╗
// ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
//
// ███████╗███╗   ██╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗
// ██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝
// █████╗  ██╔██╗ ██║██║     ██║   ██║██║  ██║██║██╔██╗ ██║██║  ███╗
// ██╔══╝  ██║╚██╗██║██║     ██║   ██║██║  ██║██║██║╚██╗██║██║   ██║
// ███████╗██║ ╚████║╚██████╗╚██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝
// ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝

#[derive(Clone, Copy, Debug)]
struct DistanceEncoder {
    multiplier: f32,
}

impl DistanceEncoder {
    fn new(edge_length: f32) -> Self {
        Self {
            multiplier: std::i8::MAX as f32 / edge_length,
        }
    }

    fn half_size(&self) -> Self {
        Self {
            multiplier: 2.0 * self.multiplier,
        }
    }

    fn encode_f32(&self, x: f32) -> i8 {
        (x * self.multiplier) as i8
    }

    fn encode_distances<D>(&self, distances: &[D; 8]) -> [i8; 8]
    where
        D: SignedDistance,
    {
        let mut encoded = [0; 8];
        for (&d, e) in distances.iter().zip(encoded.iter_mut()) {
            *e = self.encode_f32(d.into());
        }

        encoded
    }
}

#[derive(Clone, Copy, Debug)]
struct DistanceDecoder {
    multiplier: f32,
}

impl DistanceDecoder {
    fn new(edge_length: f32) -> Self {
        Self {
            multiplier: edge_length / std::i8::MAX as f32,
        }
    }

    fn half_size(&self) -> Self {
        Self {
            multiplier: 0.5 * self.multiplier,
        }
    }

    fn decode_i8(&self, x: i8) -> f32 {
        x as f32 * self.multiplier
    }

    fn decode_distances(&self, distances: &[i8; 8]) -> [f32; 8] {
        let mut decoded = [0.0; 8];
        for (&d, f) in distances.iter().zip(decoded.iter_mut()) {
            *f = self.decode_i8(d);
        }

        decoded
    }
}

// ████████╗██████╗ ██╗██╗     ██╗███╗   ██╗███████╗ █████╗ ██████╗
// ╚══██╔══╝██╔══██╗██║██║     ██║████╗  ██║██╔════╝██╔══██╗██╔══██╗
//    ██║   ██████╔╝██║██║     ██║██╔██╗ ██║█████╗  ███████║██████╔╝
//    ██║   ██╔══██╗██║██║     ██║██║╚██╗██║██╔══╝  ██╔══██║██╔══██╗
//    ██║   ██║  ██║██║███████╗██║██║ ╚████║███████╗██║  ██║██║  ██║
//    ╚═╝   ╚═╝  ╚═╝╚═╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝

fn copy_octants_to_trilinear_grid<D>(octants: &[TemporaryNode<D>]) -> [f32; 27]
where
    D: SignedDistance,
{
    /// Maps 8 octants into a 3x3x3 grid.
    const TERNARY_KERNEL: [[usize; 2]; 27] = [
        // These pairs represent ternary numbers when added together.
        [0b000, 0b000], // 0, 0t000: corner
        [0b000, 0b001], // 1, 0t001: edge
        [0b001, 0b001], // 2, 0t002: corner
        [0b000, 0b010], // 3, 0t010: edge
        [0b000, 0b011], // 4, 0t011: face
        [0b001, 0b011], // 5, 0t012: edge
        [0b010, 0b010], // 6, 0t020: corner
        [0b010, 0b011], // 7, 0t021: edge
        [0b011, 0b011], // 8, 0t022: corner
        [0b000, 0b100], // 9, 0t100: edge
        [0b000, 0b101], // 10, 0t101: face
        [0b001, 0b101], // 11, 0t102: edge
        [0b000, 0b110], // 12, 0t110: face
        [0b000, 0b111], // 13, 0t111: center
        [0b001, 0b111], // 14, 0t112: face
        [0b010, 0b110], // 15, 0t120: edge
        [0b010, 0b111], // 16, 0t121: face
        [0b011, 0b111], // 17, 0t122: edge
        [0b100, 0b100], // 18, 0t200: corner
        [0b100, 0b101], // 19, 0t201: edge
        [0b101, 0b101], // 20, 0t202: corner
        [0b100, 0b110], // 21, 0t210: edge
        [0b100, 0b111], // 22, 0t211: face
        [0b101, 0b111], // 23, 0t212: edge
        [0b110, 0b110], // 24, 0t220: corner
        [0b110, 0b111], // 25, 0t221: edge
        [0b111, 0b111], // 26, 0t222: corner
    ];
    let mut grid = [0.0; 27];
    for (d, &[o0, o1]) in grid.iter_mut().zip(TERNARY_KERNEL.iter()) {
        *d = octants[o0].distances[o1].into();
    }

    grid
}

fn trilinear_approximates_well(grid: &[f32; 27], error_tolerance: f32) -> bool {
    // This doesn't actually cover all samples, but it's a decent, fast approximation.
    const EDGES: [[usize; 3]; 12] = [
        [0, 1, 2],
        [0, 3, 6],
        [0, 9, 18],
        [2, 5, 8],
        [2, 11, 20],
        [6, 7, 8],
        [6, 15, 24],
        [8, 17, 26],
        [18, 19, 20],
        [18, 21, 24],
        [20, 23, 26],
        [24, 25, 26],
    ];
    for &[i0, i1, i2] in EDGES.iter() {
        let d0 = grid[i0];
        let d1 = grid[i1];
        let d2 = grid[i2];
        let approx_d1 = 0.5 * (d0 + d2);

        if (approx_d1 - d1).abs() > error_tolerance {
            return false;
        }
    }

    true
}

// ██╗   ██╗██╗███████╗██╗████████╗ ██████╗ ██████╗ ███████╗
// ██║   ██║██║██╔════╝██║╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
// ██║   ██║██║███████╗██║   ██║   ██║   ██║██████╔╝███████╗
// ╚██╗ ██╔╝██║╚════██║██║   ██║   ██║   ██║██╔══██╗╚════██║
//  ╚████╔╝ ██║███████║██║   ██║   ╚██████╔╝██║  ██║███████║
//   ╚═══╝  ╚═╝╚══════╝╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝

pub trait AdfOctantVisitor {
    /// Only visits leaf octants.
    fn visit_octant(&mut self, node: AdfVisitNodeId, octant: Octant, corner_distances: &[f32; 8]);
}

impl<F> AdfOctantVisitor for F
where
    F: FnMut(AdfVisitNodeId, Octant, &[f32; 8]),
{
    fn visit_octant(&mut self, node: AdfVisitNodeId, octant: Octant, corner_distances: &[f32; 8]) {
        (self)(node, octant, corner_distances)
    }
}

pub trait AdfEdgeVisitor {
    fn visit_minimal_edge(&mut self, adjacent_nodes: [AdfVisitNodeId; 4]);
}

impl<F> AdfEdgeVisitor for F
where
    F: FnMut([AdfVisitNodeId; 4]),
{
    fn visit_minimal_edge(&mut self, adjacent_nodes: [AdfVisitNodeId; 4]) {
        (self)(adjacent_nodes)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdfVisitNodeId {
    pub index: u16,
    pub is_leaf: bool,
}

impl AdfVisitNodeId {
    pub fn new(index: u16, is_leaf: bool) -> Self {
        Self { index, is_leaf }
    }

    pub fn leaf(index: u16) -> Self {
        Self {
            index,
            is_leaf: true,
        }
    }

    pub fn branch(index: u16) -> Self {
        Self {
            index,
            is_leaf: false,
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

    use crate::Array3;

    #[test]
    fn totally_negative_sdf_is_empty_adf() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let padded_extent = padded_adf_chunk_extent(&extent);
        let sdf = Array3::fill(padded_extent, -1.0);
        let adf = Adf::from_array3(&sdf, padded_extent, 1.0, 0.1);

        assert!(adf.is_empty());
    }

    #[test]
    fn totally_positive_sdf_is_empty_adf() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let padded_extent = padded_adf_chunk_extent(&extent);
        let sdf = Array3::fill(padded_extent, 1.0);
        let adf = Adf::from_array3(&sdf, padded_extent, 1.0, 0.1);

        assert!(adf.is_empty());
    }

    #[test]
    fn sphere_adf_leaves_must_contain_isosurface() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let padded_extent = padded_adf_chunk_extent(&extent);
        let radius = 8.0;
        let center = PointN([8; 3]);
        let sdf = Array3::fill_with(padded_extent, |p| (*p - center).norm() - radius);
        let adf = Adf::from_array3(&sdf, padded_extent, 1.0, 0.1);

        let mut num_visits = 0;
        let max_depth = std::usize::MAX;
        adf.visit_leaves(
            max_depth,
            &mut |_node_id: AdfVisitNodeId, _octant: Octant, distances: &[f32; 8]| {
                num_visits += 1;
                assert_contains_isosurface(distances);
            },
        );
        assert_eq!(num_visits, adf.leaves.len());
    }

    fn assert_contains_isosurface(distances: &[f32; 8]) {
        let mut num_negative = 0;
        for &d in distances.iter() {
            if d < 0.0 {
                num_negative += 1;
            }
        }
        assert!(num_negative != 0);
        assert!(num_negative != 8);
    }

    #[test]
    fn adf_compression_rate() {
        let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let padded_extent = padded_adf_chunk_extent(&extent);
        let radius = 8.0;
        let center = PointN([8; 3]);
        let sdf = Array3::fill_with(padded_extent, |p| (*p - center).norm() - radius);

        let source_size_bytes = extent.volume() as usize * std::mem::size_of::<f32>();

        for &tolerance in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0].iter() {
            let adf = Adf::from_array3(&sdf, padded_extent, 1.0, tolerance);
            let adf_size_bytes = adf.leaves.len() * std::mem::size_of::<LeafNode>()
                + adf.branches.len() * std::mem::size_of::<BranchNode>();
            test_print(&format!(
                "error tolerance = {} source SDF = {} bytes, ADF = {} bytes; rate = {:.1}%\n",
                tolerance,
                source_size_bytes,
                adf_size_bytes,
                100.0 * (adf_size_bytes as f32 / source_size_bytes as f32)
            ));
        }
    }

    fn test_print(message: &str) {
        use std::io::Write;

        std::io::stdout()
            .lock()
            .write_all(message.as_bytes())
            .unwrap();
    }
}
