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
    transforms: Query<&mut Transform>,
) {
    let target = Vec3::new(0.0, 0.0, 0.0);
    let radius = 200.0;
    let seconds = time.seconds_since_startup as f32;
    let x = radius * seconds.cos();
    let z = radius * seconds.sin();
    let eye = Vec3::new(x, 100.0, z);
    let new_transform = Mat4::face_toward(eye, target, Vec3::unit_y());

    let mut cam_tfm = transforms.get_mut(state.camera).unwrap();
    *cam_tfm = Transform::new(new_transform);
}
