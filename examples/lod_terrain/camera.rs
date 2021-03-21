use bevy::{prelude::*, render::camera::Camera};
use utilities::bevy_util::camera::create_camera_bundle;

pub fn create_camera_entity(commands: &mut Commands) -> Entity {
    commands
        .spawn(create_camera_bundle(camera_transform(0.0)))
        .current_entity()
        .unwrap()
}

pub fn camera_control_system(time: Res<Time>, mut cameras: Query<(&Camera, &mut Transform)>) {
    let mut tfm = if let Some((_camera, tfm)) = cameras.iter_mut().next() {
        tfm
    } else {
        return;
    };

    *tfm = camera_transform(time.seconds_since_startup() as f32);
}

fn camera_transform(mut t: f32) -> Transform {
    t *= 0.1;
    let position = Vec3::new(400.0 * (0.5 * t).cos(), 100.0, 400.0 * t.sin());
    let look_vec = Vec3::new(-(0.5 * t).sin(), -1.0, 2.0 * t.cos());
    let look_at = position + look_vec;

    Transform::from_translation(position).looking_at(look_at, Vec3::unit_y())
}

pub fn camera_position(t: f32) -> Vec3 {
    Vec3::new(400.0 * (0.5 * t).cos(), 100.0, 400.0 * t.sin())
}
