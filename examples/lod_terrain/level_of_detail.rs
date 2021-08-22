use crate::{
    mesh_generator::{MeshCommand, MeshCommands},
    voxel_map::VoxelMap,
};

use building_blocks::core::prelude::*;

use bevy_utilities::bevy::{prelude::*, render::camera::Camera, utils::tracing};

pub struct LodState {
    old_lod0_center: Point3f,
}

impl LodState {
    pub fn new(old_lod0_center: Point3f) -> Self {
        Self { old_lod0_center }
    }
}

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system<Map: VoxelMap>(
    cameras: Query<(&Camera, &Transform)>,
    voxel_map: Res<Map>,
    mut lod_state: ResMut<LodState>,
    mesh_commands: Res<MeshCommands>,
) {
    if mesh_commands.is_congested() {
        // Don't generate more events until the old ones have finished processing.
        return;
    }

    let lod_system_span = tracing::info_span!("lod_system");
    let _trace_guard = lod_system_span.enter();

    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    let lod0_center = Point3f::from(camera_position);

    let mut new_commands = Vec::new();
    voxel_map.clipmap_events(lod_state.old_lod0_center, lod0_center, |event| {
        new_commands.push(MeshCommand::Update(event))
    });
    mesh_commands.add_commands(new_commands.into_iter());

    lod_state.old_lod0_center = lod0_center;
}
