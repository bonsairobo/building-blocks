use crate::{voxel_map::VoxelMap, ClipSpheres, SyncBatch};

use building_blocks::prelude::LodChange3;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system(
    voxel_map: Res<VoxelMap>,
    clip_spheres: Res<ClipSpheres>,
    mesh_commands: Res<SyncBatch<LodChange3>>,
) {
    let lod_changes_span = tracing::info_span!("lod_changes");
    let mut new_commands = Vec::new();
    {
        let _trace_guard = lod_changes_span.enter();
        voxel_map.clipmap_render_updates(clip_spheres.new_sphere, |c| new_commands.push(c));
    }
    mesh_commands.extend(new_commands.into_iter());
}
