mod chunk_compressor;
mod chunk_generator;
mod clip_spheres;
mod frame_budget;
mod mesh_generator;
mod new_slot_detector;
mod sync_batch;
mod voxel_map;
mod voxel_mesh;

use chunk_compressor::chunk_compression_system;
use chunk_generator::{chunk_generator_system, GenerateBudget, GenerateTasks, NewSlot};
use clip_spheres::{clip_sphere_system, ClipSpheres};
use frame_budget::FrameBudget;
use mesh_generator::{
    mesh_deleter_system, mesh_generator_system, ChunkMeshes, MeshBudget, MeshMaterials, MeshTasks,
};
use new_slot_detector::detect_new_slots_system;
use sync_batch::SyncBatch;
use voxel_map::{MapConfig, VoxelMap};
use voxel_mesh::{BlockyMesh, SmoothMesh, VoxelMesh};

use building_blocks::{core::prelude::*, storage::prelude::clipmap_chunks_intersecting_sphere};

use bevy_utilities::{
    bevy::{
        pbr::AmbientLight,
        prelude::*,
        render::camera::{Camera, PerspectiveProjection},
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
        .add_system(chunk_generator_system.system())
        .add_system(mesh_deleter_system.system().label("mesh_deleter"))
        .add_system(mesh_generator_system::<Mesh>.system().after("mesh_deleter"))
        .add_system(chunk_compression_system.system())
        .add_system(movement_sensitivity.system())
        .add_system(light_follow_camera.system())
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

fn light_follow_camera(
    mut queries: QuerySet<(
        Query<(&Transform, &Camera)>,
        Query<(&mut Transform, &Light)>,
    )>,
) {
    let camera_position = if let Some((tfm, _cam)) = queries.q0().iter().next() {
        tfm.translation
    } else {
        return;
    };
    if let Some((mut tfm, _light)) = queries.q1_mut().iter_mut().next() {
        tfm.translation = camera_position;
    }
}

fn setup(
    map_config: Res<MapConfig>,
    async_pool: Res<ComputeTaskPool>,
    mut commands: Commands,
    mut wireframe_config: ResMut<WireframeConfig>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if map_config.wireframes {
        wireframe_config.global = true;
    }

    let map = VoxelMap::new_empty(*map_config);
    let indexer = map.chunks.indexer.clone();

    let eye = Vec3::splat(100.0);
    let clip_spheres = ClipSpheres::new(Sphere3 {
        center: Point3f::from(eye),
        radius: map_config.clip_radius,
    });

    // Detect the initial set of slots inside the clip sphere.
    let new_slots_batch = SyncBatch::<NewSlot>::default();
    let mut new_slots = Vec::new();
    clipmap_chunks_intersecting_sphere(
        &indexer,
        map_config.root_lod(),
        map_config.detect_enter_lod,
        map_config.detail,
        clip_spheres.new_sphere,
        |new_slot| new_slots.push(new_slot),
    );
    new_slots_batch.extend(new_slots.into_iter().map(|s| NewSlot { key: s.key }));

    // TODO: put these in a ChunkGenerator plugin
    commands.insert_resource(new_slots_batch);
    commands.insert_resource(MeshBudget(FrameBudget::new(
        async_pool.thread_num() as u32,
        map_config.mesh_generation_frame_time_budget_us,
        200,
    )));
    commands.insert_resource(MeshTasks::default());
    commands.insert_resource(GenerateTasks::default());
    commands.insert_resource(GenerateBudget(FrameBudget::new(
        async_pool.thread_num() as u32,
        map_config.chunk_generation_frame_time_budget_us,
        200,
    )));

    commands.insert_resource(clip_spheres);
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
        brightness: 1.0 / 2.0f32,
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 500.0, 0.0)),
        light: Light {
            intensity: 3000.0,
            depth: 0.1..100000.0,
            range: 100000.0,
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
                let mut material = StandardMaterial::from(if lod_colors {
                    colors[i % colors.len()]
                } else {
                    colors[0]
                });
                material.roughness = 0.9;

                materials.add(material)
            })
            .collect(),
    )
}
