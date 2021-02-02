use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};
use building_blocks::core::prelude::*;
use building_blocks::mesh::*;
use building_blocks::procgen::signed_distance_fields::*;
use building_blocks::storage::prelude::*;

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    commands: &mut Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let sphere_sdf = sphere(PointN([0.0; 3]), 10.0);
    let extent = Extent3i::from_min_and_max(PointN([-25; 3]), PointN([25; 3]));
    let samples = Array3::fill_with(extent, &sphere_sdf);

    let mut mesh_buffer = SurfaceNetsBuffer::default();
    let voxel_size = 1.0;
    surface_nets(&samples, samples.extent(), voxel_size, &mut mesh_buffer);
    let mesh = mesh_buffer.mesh;
    let num_vertices = mesh.positions.len();

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

    commands
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 100.0, 100.0)),
            ..Default::default()
        })
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 100.0))
                .looking_at(Vec3::zero(), Vec3::unit_y()),
            ..Default::default()
        })
        .spawn(PbrBundle {
            mesh: meshes.add(render_mesh),
            material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
            ..Default::default()
        });
}
