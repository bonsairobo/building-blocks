use bevy_utilities::bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};
use building_blocks::core::prelude::*;
use building_blocks::mesh::*;
use building_blocks::storage::prelude::*;

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    use building_blocks::core::sdfu::{self, SDF};

    let sdf = sdfu::Sphere::new(0.45)
        .subtract(sdfu::Box::new(PointN([0.25, 0.25, 1.5])))
        .union_smooth(
            sdfu::Sphere::new(0.3).translate(PointN([0.3, 0.3, 0.0])),
            0.1,
        )
        .union_smooth(
            sdfu::Sphere::new(0.3).translate(PointN([-0.3, 0.3, 0.0])),
            0.1,
        );

    let extent = Extent3i::from_min_and_max(Point3i::fill(-100), Point3i::fill(100));
    let samples = Array3x1::fill_with(extent, |p| Sd16::from(sdf.dist(0.01 * Point3f::from(p))));

    let mut mesh_buffer = SurfaceNetsBuffer::default();
    let voxel_size = 1.0;
    surface_nets(
        &samples,
        samples.extent(),
        voxel_size,
        &mut mesh_buffer,
        true,
    );
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

    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 100.0, 100.0)),
        light: Light {
            range: 200.0,
            intensity: 20000.0,
            ..Default::default()
        },
        ..Default::default()
    });
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 150.0))
            .looking_at(Vec3::new(0.0, 10.0, 0.0), Vec3::Y),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        ..Default::default()
    });
}
