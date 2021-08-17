mod blocky_voxel_map;
mod level_of_detail;
mod mesh_generator;
mod smooth_voxel_map;
mod voxel_map;

use level_of_detail::{level_of_detail_system, LodState};
use mesh_generator::{
    mesh_generator_system, ChunkMeshes, MeshCommand, MeshCommandQueue, MeshMaterials,
};
use voxel_map::{MapConfig, VoxelMap};

use building_blocks::core::prelude::*;

use bevy_inspector_egui::{Inspectable, InspectorPlugin};
use bevy_utilities::{
    bevy::{
        diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
        pbr::AmbientLight,
        prelude::*,
        render::camera::PerspectiveProjection,
        render::wireframe::{WireframeConfig, WireframePlugin},
        tasks::ComputeTaskPool,
        wgpu::{WgpuFeature, WgpuFeatures, WgpuOptions},
    },
    smooth_cameras::{controllers::fps::*, LookTransformPlugin},
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
enum Options {
    Blocky,
    Smooth,
}

fn main() {
    match Options::from_args() {
        Options::Blocky => {
            use blocky_voxel_map::BlockyVoxelMap;
            run_example::<BlockyVoxelMap>()
        }
        Options::Smooth => {
            use smooth_voxel_map::SmoothVoxelMap;
            run_example::<SmoothVoxelMap>()
        }
    }
}

fn run_example<Map: VoxelMap>() {
    let map_config = MapConfig::read_file("lod_terrain/map.ron").unwrap();

    let window_desc = WindowDescriptor {
        width: 1600.0,
        height: 900.0,
        title: "Building Blocks: LOD Terrain Example".to_string(),
        ..Default::default()
    };

    App::build()
        .insert_resource(map_config)
        .insert_resource(window_desc)
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(WgpuOptions {
            features: WgpuFeatures {
                // The Wireframe requires NonFillPolygonMode feature
                features: vec![WgpuFeature::NonFillPolygonMode],
            },
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .add_plugin(LookTransformPlugin)
        .add_plugin(FpsCameraPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(InspectorPlugin::<DiagnosticsInspectable>::new())
        .add_startup_system(setup::<Map>.system())
        .add_system(level_of_detail_system::<Map>.system())
        .add_system(mesh_generator_system::<Map>.system())
        .add_system(movement_sensitivity.system())
        .add_system(diagnostics.system())
        .run();
}

#[derive(Inspectable, Default)]
pub struct DiagnosticsInspectable {
    fps: f64,
}

fn diagnostics(
    diagnostics: Res<Diagnostics>,
    mut diagnostics_inspectable: ResMut<DiagnosticsInspectable>,
) {
    if let Some(diagnostic) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = diagnostic.value() {
            diagnostics_inspectable.fps = value;
        }
    }
}

fn movement_sensitivity(
    keyboard: Res<Input<KeyCode>>,
    mut controllers: Query<&mut FpsCameraController>,
) {
    if let Ok(mut controller) = controllers.single_mut() {
        if keyboard.pressed(KeyCode::LControl) {
            controller.translate_sensitivity = 5.;
        } else {
            controller.translate_sensitivity = 0.5;
        }
    }
}

fn setup<Map: VoxelMap>(
    map_config: Res<MapConfig>,
    mut commands: Commands,
    mut wireframe_config: ResMut<WireframeConfig>,
    pool: Res<ComputeTaskPool>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    wireframe_config.global = true;

    // Generate a voxel map from noise.
    let map = Map::generate(&*pool, *map_config);

    let eye = Vec3::splat(100.0);

    // Queue up commands to initialize the chunk meshes to their appropriate LODs given the starting camera position.
    let init_lod0_center = Point3f::from(eye);
    let mut mesh_commands = MeshCommandQueue::default();
    map.clipmap_active_chunks(init_lod0_center, |chunk_key| {
        mesh_commands.enqueue(MeshCommand::Create(chunk_key))
    });
    assert!(!mesh_commands.is_empty());
    commands.insert_resource(mesh_commands);
    commands.insert_resource(LodState::new(init_lod0_center));
    commands.insert_resource(map);
    commands.insert_resource(ChunkMeshes::default());

    commands.insert_resource(load_mesh_materials(&mut *materials));

    // Lights, camera, action!
    commands.spawn_bundle(FpsCameraBundle::new(
        FpsCameraController {
            smoothing_weight: 0.7,
            ..Default::default()
        },
        PerspectiveCameraBundle {
            perspective_projection: PerspectiveProjection {
                far: 10000.0,
                ..Default::default()
            },
            ..Default::default()
        },
        eye,
        Vec3::splat(0.0),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1.0 / 5.0f32,
    });
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
