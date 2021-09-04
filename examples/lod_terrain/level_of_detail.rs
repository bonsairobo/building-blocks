use crate::{
    mesh_generator::{MeshBudget, MeshCommands},
    voxel_map::VoxelMap,
    ClipSpheres,
};

use bevy_utilities::bevy::{prelude::*, utils::tracing};

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system(
    voxel_map: Res<VoxelMap>,
    clip_spheres: Res<ClipSpheres>,
    budget: Res<MeshBudget>,
    mut mesh_commands: ResMut<MeshCommands>,
) {
    let span = tracing::info_span!("lod_changes");
    let _trace_guard = span.enter();

    let this_frame_budget = budget.0.request_work(0);

    voxel_map.chunks.clipmap_render_updates(
        voxel_map.config.detail,
        clip_spheres.new_sphere,
        this_frame_budget as usize,
        |c| mesh_commands.push(c),
    );
}
