use bevy::{prelude::*, render::camera::PerspectiveProjection};

pub fn create_camera_bundle(transform: Transform) -> PerspectiveCameraBundle {
    PerspectiveCameraBundle {
        perspective_projection: PerspectiveProjection {
            far: 10000.0,
            ..Default::default()
        },
        transform,
        ..Default::default()
    }
}
