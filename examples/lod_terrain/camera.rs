use bevy::{
    prelude::*,
    render::camera::{Camera, PerspectiveProjection},
};

pub fn create_camera_entity(commands: &mut Commands, eye: Vec3, target: Vec3) -> Entity {
    let mut camera_components = PerspectiveCameraBundle {
        perspective_projection: PerspectiveProjection {
            far: 10000.0,
            ..Default::default()
        },
        ..Default::default()
    };
    camera_components.transform =
        Transform::from_matrix(Mat4::face_toward(eye, target, Vec3::unit_y()));

    commands.spawn(camera_components).current_entity().unwrap()
}

pub fn camera_control_system(time: Res<Time>, mut cameras: Query<(&Camera, &mut Transform)>) {
    let mut tfm = if let Some((_camera, tfm)) = cameras.iter_mut().next() {
        tfm
    } else {
        return;
    };

    let t = 0.1 * time.seconds_since_startup() as f32;

    let position = Vec3::new(400.0 * (0.5 * t).cos(), 100.0, 400.0 * t.sin());
    let look_vec = Vec3::new(-(0.5 * t).sin(), -1.0, 2.0 * t.cos());
    let look_at = position + look_vec;
    *tfm = Transform::from_translation(position).looking_at(look_at, Vec3::unit_y())
}
