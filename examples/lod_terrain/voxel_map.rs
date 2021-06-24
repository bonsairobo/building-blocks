use bevy_utilities::bevy::{ecs, tasks::ComputeTaskPool};
use building_blocks::{
    mesh::PosNormMesh,
    prelude::*,
    storage::{ChunkKey3, OctreeChunkIndex},
};

pub trait VoxelMap: ecs::component::Component {
    type MeshBuffers: Send;

    fn generate(
        pool: &ComputeTaskPool,
        freq: f32,
        scale: f32,
        seed: i32,
        octaves: u8,
        freq_heightmap: f32,
        scale_heightmap: f32,
    ) -> Self;

    fn world_chunks_extent() -> Extent3i;
    fn chunk_log2() -> i32;
    fn clip_box_radius() -> u16;
    fn world_extent() -> Extent3i;

    fn chunk_index(&self) -> &OctreeChunkIndex;

    fn init_mesh_buffers() -> Self::MeshBuffers;

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh>;
}
