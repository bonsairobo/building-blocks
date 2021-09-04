use crate::{mesh_generator::MeshCommands, voxel_map::VoxelMap, ClipSpheres};

use bevy_utilities::bevy::{prelude::*, utils::tracing};

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system(
    voxel_map: Res<VoxelMap>,
    clip_spheres: Res<ClipSpheres>,
    mut mesh_commands: ResMut<MeshCommands>,
) {
    let span = tracing::info_span!("lod_changes");
    let _trace_guard = span.enter();

    voxel_map.clipmap_render_updates(clip_spheres.new_sphere, |c| mesh_commands.push(c));
}
