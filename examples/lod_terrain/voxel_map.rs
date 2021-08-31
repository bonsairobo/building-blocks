use bevy_utilities::{
    bevy::tasks::ComputeTaskPool,
    noise::{generate_noise_chunk3, generate_noise_chunks3},
};
use building_blocks::{
    mesh::{IsOpaque, MergeVoxel},
    prelude::*,
};

use serde::{Deserialize, Serialize};

pub struct VoxelMap {
    pub config: MapConfig,
    pub chunks: HashMapChunkTree3x1<Voxel>,
}

impl VoxelMap {
    pub fn generate_lod0(pool: &ComputeTaskPool, config: MapConfig) -> Self {
        let MapConfig {
            chunk_exponent,
            num_lods,
            world_chunks_extent,
            noise:
                NoiseConfig {
                    freq,
                    scale,
                    seed,
                    octaves,
                },
            ..
        } = config;

        let chunk_shape = Point3i::fill(1 << chunk_exponent);

        // let noise_chunks = generate_noise_chunks3(
        //     pool,
        //     world_chunks_extent,
        //     chunk_shape,
        //     freq,
        //     scale,
        //     seed,
        //     octaves,
        //     true,
        // );

        let root_lod = num_lods - 1;
        let builder = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
            chunk_shape,
            ambient_value: Voxel::EMPTY,
            root_lod,
        });
        let mut chunks = builder.build_with_hash_map_storage();

        // for (chunk_min, noise) in noise_chunks.into_iter() {
        //     chunks.write_chunk(ChunkKey::new(0, chunk_min), unsafe {
        //         // SAFE: Voxel is a transparent wrapper of f32
        //         std::mem::transmute(noise)
        //     });
        // }

        Self { chunks, config }
    }

    pub fn clipmap_render_updates(&self, new_clip_sphere: Sphere3, rx: impl FnMut(LodChange3)) {
        self.chunks.clipmap_render_updates(
            self.config.detail,
            new_clip_sphere,
            self.config.chunks_processed_per_frame,
            rx,
        );
    }

    pub fn clipmap_loading_slots(&self, clip_sphere: Sphere3, rx: impl FnMut(ChunkKey3)) {
        self.chunks
            .clipmap_loading_slots(self.config.chunks_processed_per_frame, clip_sphere, rx);
    }

    pub fn generate_lod0_chunk(config: &MapConfig, chunk_min: Point3i) -> Option<Array3x1<Voxel>> {
        let chunk_shape = config.chunk_shape();

        let NoiseConfig {
            freq,
            scale,
            seed,
            octaves,
        } = config.noise;

        let chunk_extent = Extent3i::from_min_and_shape(chunk_min, chunk_shape);

        unsafe {
            // SAFE: Voxel is a transparent wrapper of f32
            std::mem::transmute(generate_noise_chunk3(
                chunk_extent,
                freq,
                scale,
                seed,
                octaves,
                true,
            ))
        }
    }

    pub fn downsample_all(&mut self, pool: &ComputeTaskPool) {
        let chunks = &self.chunks;
        let new_chunks = pool.scope(|scope| {
            chunks.visit_root_keys(|root| {
                scope.spawn(async move {
                    let mut downsampled_chunks = Vec::new();
                    chunks.downsample_descendants_into_new_chunks(
                        &SdfMeanDownsampler,
                        root,
                        0,
                        |key, new_chunk| downsampled_chunks.push((key, new_chunk)),
                    );
                    downsampled_chunks
                });
            });
        });
        for root_new_chunks in new_chunks.into_iter() {
            for (key, new_chunk) in root_new_chunks.into_iter() {
                self.chunks.write_chunk(key, new_chunk);
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

#[derive(Copy, Clone, PartialEq)]
pub struct Voxel(pub f32);

impl Voxel {
    pub const EMPTY: Self = Self(1.0);
    pub const FILLED: Self = Self(-1.0);
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        self.0 >= 0.0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl MergeVoxel for Voxel {
    type VoxelValue = bool;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.0 < 0.0
    }
}

impl From<Voxel> for f32 {
    fn from(v: Voxel) -> Self {
        v.0
    }
}

impl From<f32> for Voxel {
    fn from(x: f32) -> Self {
        Voxel(x)
    }
}

impl SignedDistance for Voxel {
    fn is_negative(&self) -> bool {
        self.0.is_negative()
    }
}
