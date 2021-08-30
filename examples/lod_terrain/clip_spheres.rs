use crate::voxel_map::MapConfig;

use bevy_utilities::bevy::{prelude::*, render::camera::Camera};

use building_blocks::core::prelude::*;

pub fn clip_sphere_system(
    config: Res<MapConfig>,
    cameras: Query<(&Camera, &Transform)>,
    mut clip_spheres: ResMut<ClipSpheres>,
) {
    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    clip_spheres.old_sphere = clip_spheres.new_sphere;
    clip_spheres.new_sphere = Sphere3 {
        center: Point3f::from(camera_position),
        radius: config.clip_radius,
    };
}

pub struct ClipSpheres {
    pub old_sphere: Sphere3,
    pub new_sphere: Sphere3,
}

impl ClipSpheres {
    pub fn new(sphere: Sphere3) -> Self {
        Self {
            old_sphere: sphere,
            new_sphere: sphere,
        }
    }
}
