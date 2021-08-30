use crate::{
    mesh_generator::{MeshCommand, MeshCommands},
    voxel_map::VoxelMap,
    ClipSpheres,
};

use bevy_utilities::bevy::{prelude::*, utils::tracing};

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system<Map: VoxelMap>(
    voxel_map: Res<Map>,
    clip_spheres: Res<ClipSpheres>,
    mesh_commands: Res<MeshCommands>,
) {
    let lod_changes_span = tracing::info_span!("lod_changes");
    let mut new_commands = Vec::new();
    {
        let _trace_guard = lod_changes_span.enter();
        voxel_map.clipmap_render_updates(clip_spheres.new_sphere, |c| {
            new_commands.push(MeshCommand::LodChange(c))
        });
    }
    mesh_commands.add_commands(new_commands.into_iter());
}
