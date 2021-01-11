use crate::{interpolate_position_from_sdf_cube, normal_from_sdf_cube, PosNormMesh};

use building_blocks_core::prelude::*;
use building_blocks_storage::{Adf, AdfVisitNodeId, Octant};

#[derive(Default)]
pub struct AdfDualContourBuffer {
    /// The isosurface positions and normals. The normals are *not* normalized, since that is done most efficiently on the GPU.
    pub mesh: PosNormMesh,

    // Used to map back from an ADF node to vertex index.
    branch_to_index: Vec<u32>,
    leaf_to_index: Vec<u32>,
}

impl AdfDualContourBuffer {
    pub fn reset(&mut self) {
        self.mesh.clear();
        self.branch_to_index.clear();
        self.leaf_to_index.clear();
    }
}

/// Generates a hexahedral mesh of the dual graph of the adaptive distance field. This should approximate the isosurface, giving
/// better approximations with greater `max_depth`.
pub fn adf_dual_contour(adf: &Adf, output: &mut AdfDualContourBuffer) {
    output.reset();
    output.branch_to_index.resize(adf.num_branches(), 0);
    output.leaf_to_index.resize(adf.num_leaves(), 0);

    let max_depth = std::usize::MAX;
    adf.visit_leaves(
        max_depth,
        &mut |node_id: AdfVisitNodeId, octant: Octant, distances: &[f32; 8]| {
            let position = Point3f::from(octant.minimum())
                + interpolate_position_from_sdf_cube(distances) * octant.edge_length() as f32;
            let normal = normal_from_sdf_cube(distances);

            let vertex_index = output.mesh.positions.len() as u32;
            if node_id.is_leaf {
                output.leaf_to_index[node_id.index as usize] = vertex_index;
            } else {
                output.branch_to_index[node_id.index as usize] = vertex_index;
            };
            output.mesh.positions.push(position.0);
            output.mesh.normals.push(normal);
        },
    );

    adf.visit_minimal_edges(&mut |quad_nodes: [AdfVisitNodeId; 4]| {
        let mut quad_indices = [0; 4];
        for (node, i) in quad_nodes.iter().zip(quad_indices.iter_mut()) {
            if node.is_leaf {
                *i = output.leaf_to_index[node.index as usize];
            } else {
                *i = output.branch_to_index[node.index as usize];
            }
        }

        // Counter-clockwise triangles, assuming nodes are in Z order.
        output.mesh.indices.push(quad_indices[0]);
        output.mesh.indices.push(quad_indices[2]);
        output.mesh.indices.push(quad_indices[1]);
        output.mesh.indices.push(quad_indices[1]);
        output.mesh.indices.push(quad_indices[2]);
        output.mesh.indices.push(quad_indices[3]);
    });
}
