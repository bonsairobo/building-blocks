use bevy_utilities::bevy::{ecs, tasks::ComputeTaskPool};
use building_blocks::{mesh::PosNormMesh, prelude::*};

use serde::{Deserialize, Serialize};

pub trait VoxelMap: ecs::component::Component {
    type MeshBuffers: Send;

    type Chunk: Send;

    fn generate_lod0(pool: &ComputeTaskPool, config: MapConfig) -> Self;

    fn downsample_descendants_into_new_chunks(
        &self,
        node_key: ChunkKey3,
        chunk_rx: impl FnMut(ChunkKey3, Self::Chunk),
    );

    fn write_chunk(&mut self, key: ChunkKey3, chunk: Self::Chunk);

    fn config(&self) -> &MapConfig;

    fn clipmap_render_updates(&self, new_clip_sphere: Sphere3, rx: impl FnMut(LodChange3));

    fn chunk_indexer(&self) -> &ChunkIndexer3;

    fn root_lod(&self) -> u8;

    fn visit_root_keys(&self, visitor: impl FnMut(ChunkKey3));

    fn init_mesh_buffers(&self) -> Self::MeshBuffers;

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh>;

    fn downsample_all(&mut self, pool: &ComputeTaskPool) {
        let self_ref = &*self;
        let new_chunks = pool.scope(|scope| {
            self_ref.visit_root_keys(|root| {
                scope.spawn(async move {
                    let mut downsampled_chunks = Vec::new();
                    self_ref.downsample_descendants_into_new_chunks(root, |key, new_chunk| {
                        downsampled_chunks.push((key, new_chunk))
                    });
                    downsampled_chunks
                });
            });
        });
        for root_new_chunks in new_chunks.into_iter() {
            for (key, new_chunk) in root_new_chunks.into_iter() {
                self.write_chunk(key, new_chunk);
            }
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct MapConfig {
    pub chunk_exponent: u8,
    pub num_lods: u8,
    pub clip_radius: f32,
    pub detect_enter_lod: u8,
    pub detail: f32,
    pub chunks_processed_per_frame: usize,
    pub world_chunks_extent: ChunkUnits<Extent3i>,
    pub noise: NoiseConfig,
    pub wireframes: bool,
    pub lod_colors: bool,
    pub msaa: Option<u32>,
}

impl MapConfig {
    pub fn read_file(path: &str) -> Result<Self, ron::Error> {
        let reader = std::fs::File::open(path)?;

        ron::de::from_reader(reader)
    }

    pub fn chunk_shape(&self) -> Point3i {
        Point3i::fill(1 << self.chunk_exponent)
    }

    pub fn root_lod(&self) -> u8 {
        self.num_lods - 1
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct NoiseConfig {
    pub freq: f32,
    pub scale: f32,
    pub seed: i32,
    pub octaves: u8,
}
