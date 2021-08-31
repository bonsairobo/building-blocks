mod chunk_generator;
mod clip_spheres;
mod level_of_detail;
mod mesh_generator;
mod new_slot_detector;
mod sync_batch;
mod voxel_map;
mod voxel_mesh;

use chunk_generator::{
    chunk_downsampler_system, chunk_generator_system, find_loading_slots_system,
    new_chunk_writer_system, DownsampleSlots, GenerateSlots, LoadedChunks, NewSlot,
};
use clip_spheres::{clip_sphere_system, ClipSpheres};
use level_of_detail::level_of_detail_system;
use mesh_generator::{mesh_deleter_system, mesh_generator_system, ChunkMeshes, MeshMaterials};
use new_slot_detector::detect_new_slots_system;
use sync_batch::SyncBatch;
use voxel_map::{MapConfig, VoxelMap};
use voxel_mesh::{BlockyMesh, SmoothMesh, VoxelMesh};

use building_blocks::{core::prelude::*, prelude::LodChange3};

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
        Options::Blocky => run_example::<BlockyMesh>(),
        Options::Smooth => run_example::<SmoothMesh>(),
    }
}

fn run_example<Mesh: VoxelMesh>() {
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
        .add_startup_system(setup.system())
        .add_system(clip_sphere_system.system())
        .add_system(detect_new_slots_system.system())
        // TODO: put these in a ChunkGenerator plugin
        // We need a specific ordering for these systems because we want to ensure that the chunks we found to load are written
        // back to the chunk tree in the same frame.
        .add_system(find_loading_slots_system.system().label("find_loading"))
        .add_system(
            chunk_generator_system
                .system()
                .after("find_loading")
                .before("new_chunk_writer"),
        )
        .add_system(
            chunk_downsampler_system
                .system()
                .after("find_loading")
                .before("new_chunk_writer"),
        )
        .add_system(
            new_chunk_writer_system
                .system()
                .label("new_chunk_writer")
                .after("find_loading"),
        )
        .add_system(level_of_detail_system.system())
        .add_system(mesh_deleter_system.system().label("mesh_deleter"))
        .add_system(mesh_generator_system::<Mesh>.system().after("mesh_deleter"))
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

fn setup(
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
    let mut map = VoxelMap::generate_lod0(&*pool, *map_config);
    map.downsample_all(&*pool);

    let eye = Vec3::splat(100.0);

    // TODO: put these in a ChunkGenerator plugin
    commands.insert_resource(SyncBatch::<LodChange3>::default());
    commands.insert_resource(SyncBatch::<NewSlot>::default());
    commands.insert_resource(GenerateSlots::default());
    commands.insert_resource(DownsampleSlots::default());
    commands.insert_resource(LoadedChunks::default());

    commands.insert_resource(ClipSpheres::new(Sphere3 {
        center: Point3f::from(eye),
        radius: map_config.clip_radius,
    }));
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
