use bevy_utilities::bevy::{ecs, tasks::ComputeTaskPool};
use building_blocks::{mesh::PosNormMesh, prelude::*};

use serde::{Deserialize, Serialize};

pub trait VoxelMap: ecs::component::Component {
    type MeshBuffers: Send;

    fn generate(pool: &ComputeTaskPool, config: MapConfig) -> Self;

    fn config(&self) -> &MapConfig;

    fn chunk_index(&self) -> &OctreeChunkIndex;

    fn init_mesh_buffers(&self) -> Self::MeshBuffers;

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh>;
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct MapConfig {
    pub superchunk_shape: Point3i,
    pub chunk_shape: Point3i,
    pub num_lods: u8,
    pub clip_box_radius: u16,
    pub world_chunks_extent: ChunkUnits<Extent3i>,
    pub noise: NoiseConfig,
}

impl MapConfig {
    pub fn read_file(path: &str) -> Result<Self, ron::Error> {
        let reader = std::fs::File::open(path)?;

        ron::de::from_reader(reader)
    }

    pub fn world_extent(&self) -> Extent3i {
        self.world_chunks_extent.0 * self.chunk_shape
    }

    pub fn chunk_log2(&self) -> i32 {
        self.chunk_shape.x().trailing_zeros() as i32
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct NoiseConfig {
    pub freq: f32,
    pub scale: f32,
    pub seed: i32,
    pub octaves: u8,
}
