use crate::{
    mesh_generator::{MeshCommand, MeshCommandQueue},
    voxel_map::{VoxelMap, CHUNK_LOG2, CLIP_BOX_RADIUS, WORLD_EXTENT},
};

use building_blocks::core::prelude::*;

use bevy::{prelude::*, render::camera::Camera};

#[derive(Default)]
pub struct LodState {
    old_lod0_center: Point3i,
}

impl LodState {
    pub fn new(lod0_center: Point3i) -> Self {
        LodState {
            old_lod0_center: lod0_center,
        }
    }
}

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system(
    cameras: Query<(&Camera, &Transform)>,
    voxel_map: Res<VoxelMap>,
    mut lod_state: ResMut<LodState>,
    mut mesh_commands: ResMut<MeshCommandQueue>,
) {
    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    let lod0_center = Point3f::from(camera_position).in_voxel() >> CHUNK_LOG2;

    if lod0_center == lod_state.old_lod0_center {
        return;
    }

    voxel_map.index.find_clipmap_chunk_updates(
        &WORLD_EXTENT,
        CLIP_BOX_RADIUS,
        lod_state.old_lod0_center,
        lod0_center,
        |update| mesh_commands.enqueue(MeshCommand::Update(update)),
    );

    lod_state.old_lod0_center = lod0_center;
}
