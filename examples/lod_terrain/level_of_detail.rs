use crate::{
    mesh_generator::{MeshCommand, MeshCommands},
    voxel_map::VoxelMap,
};

use building_blocks::{core::prelude::*, storage::chunk_tree::LodChange};

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
    time: Res<Time>,
) {
    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    let new_lod0_center = Point3f::from(camera_position);

    let old_clip_sphere = Sphere3 {
        center: lod_state.old_lod0_center,
        radius: voxel_map.config().clip_radius,
    };
    let new_clip_sphere = Sphere3 {
        center: new_lod0_center,
        radius: voxel_map.config().clip_radius,
    };

    let mut new_commands = Vec::new();
    let clip_events_span = tracing::info_span!("clip_events");
    let lod_changes_span = tracing::info_span!("lod_changes");
    {
        let _trace_guard = clip_events_span.enter();
        voxel_map.clipmap_new_chunks(old_clip_sphere, new_clip_sphere, |new_chunk| {
            // TODO: handle min LOD entrances
            if new_chunk.is_active {
                new_commands.push(MeshCommand::Create(new_chunk.key));
            }
        });
    }
    {
        let _trace_guard = lod_changes_span.enter();
        voxel_map.clipmap_render_updates(new_clip_sphere, |c| {
            new_commands.push(MeshCommand::LodChange(c))
        });
    }
    mesh_commands.add_commands(new_commands.into_iter());

    lod_state.old_lod0_center = new_lod0_center;
}
