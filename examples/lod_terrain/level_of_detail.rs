use crate::{
    chunk_generator::{ChunkCommand, ChunkCommandQueue},
    mesh_generator::{MeshCommand, MeshCommandQueue},
    voxel_map::{MapConfig, VoxelMap},
};

use building_blocks::{core::prelude::*, storage::prelude::ClipEvent3};

use bevy_utilities::bevy::{prelude::*, render::camera::Camera};

pub struct LodState {
    old_lod0_center: Point3f,
    lod0_center: Point3f,
}

impl LodState {
    pub fn new(lod0_center: Point3f) -> Self {
        Self {
            old_lod0_center: lod0_center,
            lod0_center,
        }
    }
}

pub fn level_of_detail_state_update_system(
    cameras: Query<(&Camera, &Transform)>,
    mut lod_state: ResMut<LodState>,
) {
    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    let lod0_center = Point3f::from(camera_position);

    lod_state.old_lod0_center = lod_state.lod0_center;
    lod_state.lod0_center = lod0_center;
}

/// Adjusts the sample rate of voxels depending on their distance from the camera.
pub fn level_of_detail_system<Map: VoxelMap>(
    voxel_map: Res<Map>,
    lod_state: Res<LodState>,
    mut chunk_commands: ResMut<ChunkCommandQueue>,
    mut mesh_commands: ResMut<MeshCommandQueue>,
) {
    voxel_map.clipmap_events(
        lod_state.old_lod0_center,
        lod_state.lod0_center,
        |update| match update {
            ClipEvent3::Enter(chunk_key, is_active_lod) => {
                let MapConfig {
                    chunk_exponent,
                    world_chunks_extent,
                    ..
                } = *voxel_map.config();
                let world_chunks_extent_lod0 = world_chunks_extent.0 << chunk_exponent;
                let lod0_extent = voxel_map.chunk_extent_at_lower_lod(chunk_key, 0);

                // Only generate chunks at lod 0 and within the world extent y-height
                if chunk_key.lod == 0
                    && lod0_extent.minimum.y() >= world_chunks_extent_lod0.minimum.y()
                    && lod0_extent.minimum.y() <= world_chunks_extent_lod0.max().y()
                    && lod0_extent.max().y() >= world_chunks_extent_lod0.minimum.y()
                    && lod0_extent.max().y() <= world_chunks_extent_lod0.max().y()
                {
                    chunk_commands.enqueue(ChunkCommand::Create(chunk_key));
                }

                if (lod0_extent.minimum.y() >= world_chunks_extent_lod0.minimum.y()
                    && lod0_extent.minimum.y() <= world_chunks_extent_lod0.max().y())
                    || (lod0_extent.max().y() >= world_chunks_extent_lod0.minimum.y()
                        && lod0_extent.max().y() <= world_chunks_extent_lod0.max().y())
                {
                    // Only downsample lods > 0 that intersect the world extent
                    if chunk_key.lod > 0 {
                        chunk_commands.enqueue(ChunkCommand::Downsample(chunk_key));
                    }
                    // Only mesh if the lod is active, and the extent intersects the world extent
                    if is_active_lod {
                        mesh_commands.enqueue(MeshCommand::Create(chunk_key));
                    }
                }
            }
            ClipEvent3::Exit(chunk_key, was_active_lod) => {
                if was_active_lod {
                    chunk_commands.enqueue(ChunkCommand::Destroy(chunk_key));
                    mesh_commands.enqueue(MeshCommand::Destroy(chunk_key));
                }
            }
            ClipEvent3::Split(_) | ClipEvent3::Merge(_) => {
                mesh_commands.enqueue(MeshCommand::Update(update))
            }
        },
    );
}
