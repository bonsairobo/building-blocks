use building_blocks_mesh::PosNormMesh;

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};

pub fn create_mesh_bundle(
    mut mesh: PosNormMesh,
    material: Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
) -> PbrBundle {
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    let num_vertices = mesh.positions.len();

    // Bevy might not normalize our surface normals in the vertex shader (before interpolation).
    for n in &mut mesh.normals {
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        n[0] = n[0] / norm;
        n[1] = n[1] / norm;
        n[2] = n[2] / norm;
    }

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(mesh.positions),
    );
    render_mesh.set_attribute("Vertex_Normal", VertexAttributeValues::Float3(mesh.normals));
    render_mesh.set_attribute(
        "Vertex_Uv",
        VertexAttributeValues::Float2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(mesh.indices)));

    PbrBundle {
        mesh: meshes.add(render_mesh),
        material,
        ..Default::default()
    }
}
