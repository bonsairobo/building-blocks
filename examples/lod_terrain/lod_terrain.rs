mod blocky_voxel_map;
mod level_of_detail;
mod mesh_generator;
mod smooth_voxel_map;
mod voxel_map;

use level_of_detail::{level_of_detail_system, LodState};
use mesh_generator::{
    mesh_deleter_system, mesh_generator_system, ChunkMeshes, MeshCommands, MeshMaterials,
};
use voxel_map::{MapConfig, VoxelMap};

use building_blocks::core::prelude::*;

use bevy_utilities::{
    bevy::{
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

    let mut builder = App::build();
    builder
        .insert_resource(map_config)
        .insert_resource(window_desc)
        .insert_resource(WgpuOptions {
            features: WgpuFeatures {
                // The Wireframe requires NonFillPolygonMode feature
                features: vec![WgpuFeature::NonFillPolygonMode],
            },
            ..Default::default()
        });

    if let Some(samples) = map_config.msaa {
        builder.insert_resource(Msaa { samples });
    }

    builder
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .add_plugin(LookTransformPlugin)
        .add_plugin(FpsCameraPlugin)
        .add_startup_system(setup::<Map>.system())
        .add_system(level_of_detail_system::<Map>.system())
        .add_system(mesh_deleter_system::<Map>.system().label("mesh_deleter"))
        .add_system(mesh_generator_system::<Map>.system().after("mesh_deleter"))
        .add_system(movement_sensitivity.system())
        .run();
}

fn movement_sensitivity(
    keyboard: Res<Input<KeyCode>>,
    mut controllers: Query<&mut FpsCameraController>,
) {
    if let Ok(mut controller) = controllers.single_mut() {
        if keyboard.pressed(KeyCode::LControl) {
            controller.translate_sensitivity = 20.;
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
    if map_config.wireframes {
        wireframe_config.global = true;
    }

    // Generate a voxel map from noise.
    let map = Map::generate(&*pool, *map_config);

    let eye = Vec3::splat(100.0);

    commands.insert_resource(MeshCommands::default());
    commands.insert_resource(LodState::new(Point3f::from(eye)));
    commands.insert_resource(map);
    commands.insert_resource(ChunkMeshes::default());

    commands.insert_resource(load_mesh_materials(map_config.lod_colors, &mut *materials));

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

fn load_mesh_materials(
    lod_colors: bool,
    materials: &mut Assets<StandardMaterial>,
) -> MeshMaterials {
    let colors = [
        Color::rgb(1.0, 0.0, 0.0),
        Color::rgb(0.0, 1.0, 0.0),
        Color::rgb(0.0, 0.0, 1.0),
        Color::rgb(1.0, 1.0, 0.0),
        Color::rgb(0.0, 1.0, 1.0),
        Color::rgb(1.0, 0.0, 1.0),
    ];

    MeshMaterials(
        (0..6)
            .map(|i| {
                let mut material =
                    StandardMaterial::from(if lod_colors { colors[i] } else { colors[0] });
                material.roughness = 0.9;

                materials.add(material)
            })
            .collect(),
    )
}
