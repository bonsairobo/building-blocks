use bevy_utilities::bevy::{ecs, tasks::ComputeTaskPool};
use building_blocks::{mesh::PosNormMesh, prelude::*};

use serde::{Deserialize, Serialize};

pub trait VoxelMap: ecs::component::Component {
    type Voxel: Send;
    type MeshBuffers: Send;

    fn generate(pool: &ComputeTaskPool, config: MapConfig) -> Self;

    fn chunk_extent_at_lower_lod(&self, key: ChunkKey3, lod: u8) -> Extent3i;

    fn iter_chunks_for_key(&self, key: ChunkKey3) -> Box<dyn Iterator<Item = Point3i>>;

    fn chunk_is_generated(&self, minimum: Point3i, lod: u8) -> bool;

    fn generate_chunk(&self, minimum: Point3i) -> Option<Array3x1<Self::Voxel>>;

    fn write_chunk(&mut self, key: ChunkKey3, chunk: Array3x1<Self::Voxel>);

    fn remove_chunk(&mut self, key: ChunkKey3);

    fn downsample_extent_into_self(&mut self, extent: Extent3i, src_lod: u8, max_lod: u8);

    fn config(&self) -> &MapConfig;

    fn clipmap_active_chunks(&self, lod0_center: Point3f, active_rx: impl FnMut(ChunkKey3));

    fn clipmap_events(
        &self,
        old_lod0_center: Point3f,
        new_lod0_center: Point3f,
        update_rx: impl FnMut(ClipEvent3),
    );

    fn init_mesh_buffers(&self) -> Self::MeshBuffers;

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh>;
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct MapConfig {
    pub chunk_exponent: u8,
    pub num_lods: u8,
    pub clip_radius: f32,
    pub detail: f32,
    pub world_chunks_extent: ChunkUnits<Extent3i>,
    pub noise: NoiseConfig,
    pub wireframes: bool,
    pub msaa: Option<u32>,
}

impl MapConfig {
    pub fn read_file(path: &str) -> Result<Self, ron::Error> {
        let reader = std::fs::File::open(path)?;

        ron::de::from_reader(reader)
    }

    pub fn world_extent(&self) -> Extent3i {
        self.world_chunks_extent.0 * self.chunk_shape()
    }

    pub fn chunk_shape(&self) -> Point3i {
        Point3i::fill(1 << self.chunk_exponent)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct NoiseConfig {
    pub freq: f32,
    pub scale: f32,
    pub seed: i32,
    pub octaves: u8,
}
