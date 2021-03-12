use bevy::prelude::*;

pub struct CameraRotationState {
    camera: Entity,
}

impl CameraRotationState {
    pub fn new(camera: Entity) -> Self {
        Self { camera }
    }
}

pub fn camera_rotation_system(
    state: Res<CameraRotationState>,
    time: Res<Time>,
    mut transforms: Query<&mut Transform>,
) {
    let t = 0.3 * time.seconds_since_startup() as f32;

    let target = Vec3::new(0.0, 0.0, 0.0);
    let height = 30.0 * (2.0 * t).sin();
    let radius = 90.0;
    let x = radius * t.cos();
    let z = radius * t.sin();
    let eye = Vec3::new(x, height, z);
    let new_transform = Mat4::face_toward(eye, target, Vec3::unit_y());

    let mut cam_tfm = transforms.get_mut(state.camera).unwrap();
    *cam_tfm = Transform::from_matrix(new_transform);
}
