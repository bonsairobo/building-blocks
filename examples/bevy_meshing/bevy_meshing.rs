mod camera_rotation;
mod mesh_generator;

use camera_rotation::{camera_rotation_system, CameraRotationState};
use mesh_generator::{mesh_generator_system, MeshGeneratorState, MeshMaterial};

use bevy::prelude::*;

fn main() {
    let mut window_desc = WindowDescriptor::default();
    window_desc.width = 1600.;
    window_desc.height = 900.;
    window_desc.title = "Building Blocks: Bevy Meshing Example".to_string();

    App::build()
        .add_resource(window_desc)
        .add_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_resource(ClearColor(Color::rgb(0.3, 0.3, 0.3)))
        .add_startup_system(setup.system())
        .add_system(camera_rotation_system.system())
        .add_system(mesh_generator_system.system())
        .run();
}

fn setup(commands: &mut Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.100, 150.0, 100.0)),
        ..Default::default()
    });

    let camera_entity = commands
        .spawn(Camera3dBundle::default())
        .current_entity()
        .unwrap();

    commands
        .insert_resource(CameraRotationState::new(camera_entity))
        .insert_resource(MeshMaterial(
            materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        ))
        .insert_resource(MeshGeneratorState::new());
}
