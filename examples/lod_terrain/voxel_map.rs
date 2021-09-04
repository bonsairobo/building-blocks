use bevy_utilities::noise::generate_noise_chunk3;
use building_blocks::{
    core::bytemuck::{Pod, Zeroable},
    mesh::{IsOpaque, MergeVoxel},
    prelude::*,
};

use serde::{Deserialize, Serialize};

pub struct VoxelMap {
    pub config: MapConfig,
    pub chunks: CompressibleChunkTree3x1<Lz4, Voxel>,
}

impl VoxelMap {
    pub fn new_empty(config: MapConfig) -> Self {
        let MapConfig {
            chunk_exponent,
            num_lods,
            ..
        } = config;

        let chunk_shape = Point3i::fill(1 << chunk_exponent);

        let root_lod = num_lods - 1;
        let builder = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
            chunk_shape,
            ambient_value: Voxel::EMPTY,
            root_lod,
        });
        let chunks = builder.build_with_storage(|| {
            FastCompressibleChunkStorageNx1::with_bytes_compression(Lz4 { level: 10 })
        });

        Self { chunks, config }
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
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct MapConfig {
    pub chunk_exponent: u8,
    pub num_lods: u8,
    pub clip_radius: f32,
    pub detect_enter_lod: u8,
    pub detail: f32,
    pub target_frame_processing_time_us: u32,
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

unsafe impl Zeroable for Voxel {}
unsafe impl Pod for Voxel {}

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
