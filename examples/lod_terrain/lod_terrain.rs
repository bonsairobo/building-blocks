mod camera;
mod level_of_detail;
mod mesh_generator;
mod voxel_map;

use camera::{camera_control_system, camera_position, create_camera_entity};
use level_of_detail::{level_of_detail_system, LodState};
use mesh_generator::{
    mesh_generator_system, ChunkMeshes, MeshCommand, MeshCommandQueue, MeshMaterials,
};
use voxel_map::{generate_map, world_extent, CHUNK_LOG2, CLIP_BOX_RADIUS, WORLD_CHUNKS_EXTENT};

use building_blocks::core::prelude::*;

use bevy::{
    prelude::*,
    // render::wireframe::{WireframeConfig, WireframePlugin},
    tasks::ComputeTaskPool,
    // wgpu::{WgpuFeature, WgpuFeatures, WgpuOptions},
};

fn main() {
    let mut window_desc = WindowDescriptor::default();
    window_desc.width = 1600.0;
    window_desc.height = 900.0;
    window_desc.title = "Building Blocks: LOD Terrain Example".to_string();

    App::build()
        .insert_resource(window_desc)
        .insert_resource(Msaa { samples: 4 })
        // .insert_resource(WgpuOptions {
        //     features: WgpuFeatures {
        //         // The Wireframe requires NonFillPolygonMode feature
        //         features: vec![WgpuFeature::NonFillPolygonMode],
        //     },
        //     ..Default::default()
        // })
        .add_plugins(DefaultPlugins)
        // .add_plugin(WireframePlugin)
        .add_startup_system(setup.system())
        .add_system(level_of_detail_system.system())
        .add_system(mesh_generator_system.system())
        .add_system(camera_control_system.system())
        .run();
}

fn setup(
    mut commands: Commands,
    // mut wireframe_config: ResMut<WireframeConfig>,
    pool: Res<ComputeTaskPool>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // wireframe_config.global = true;

    // Generate a voxel map from noise.
    let freq = 0.15;
    let scale = 20.0;
    let seed = 666;
    let map = generate_map(&*pool, WORLD_CHUNKS_EXTENT, freq, scale, seed);

    // Queue up commands to initialize the chunk meshes to their appropriate LODs given the starting camera position.
    let init_lod0_center = Point3f::from(camera_position(0.0)).in_voxel() >> CHUNK_LOG2;
    let mut mesh_commands = MeshCommandQueue::default();
    map.index.active_clipmap_lod_chunks(
        &world_extent(),
        CLIP_BOX_RADIUS,
        init_lod0_center,
        |chunk_key| mesh_commands.enqueue(MeshCommand::Create(chunk_key)),
    );
    assert!(!mesh_commands.is_empty());
    commands.insert_resource(mesh_commands);
    commands.insert_resource(LodState::new(init_lod0_center));
    commands.insert_resource(map);
    commands.insert_resource(ChunkMeshes::default());

    commands.insert_resource(load_mesh_materials(&mut *materials));

    // Lights, camera, action!
    create_camera_entity(&mut commands);
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 500.0, 0.0)),
        light: Light {
            intensity: 1000000.0,
            depth: 0.1..1000000.0,
            range: 1000000.0,
            ..Default::default()
        },
        ..Default::default()
    });
}

fn load_mesh_materials(materials: &mut Assets<StandardMaterial>) -> MeshMaterials {
    let colors = [
        Color::rgb(1.0, 0.0, 0.0),
        Color::rgb(0.0, 1.0, 0.0),
        Color::rgb(0.0, 0.0, 1.0),
        Color::rgb(1.0, 1.0, 0.0),
        Color::rgb(0.0, 1.0, 1.0),
        Color::rgb(1.0, 0.0, 1.0),
    ];

    MeshMaterials(
        colors
            .iter()
            .map(|c| {
                let mut material = StandardMaterial::from(*c);
                material.roughness = 0.9;

                materials.add(material)
            })
            .collect(),
    )
}
