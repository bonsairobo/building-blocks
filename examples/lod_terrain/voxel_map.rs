use bevy::tasks::ComputeTaskPool;
use building_blocks::{
    prelude::*,
    storage::{ChunkHashMapPyramid3, OctreeChunkIndex, SmallKeyHashMap},
};

use building_blocks_mesh::{IsOpaque, MergeVoxel};
use simdnoise::NoiseBuilder;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Voxel(pub u8);

impl Voxel {
    pub const EMPTY: Self = Self(0);
    pub const FILLED: Self = Self(1);
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl MergeVoxel for Voxel {
    type VoxelValue = u8;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.0
    }
}

pub struct VoxelMap {
    pub pyramid: ChunkHashMapPyramid3<Voxel>,
    pub index: OctreeChunkIndex,
}

pub fn generate_map(
    pool: &ComputeTaskPool,
    chunks_extent: Extent3i,
    freq: f32,
    scale: f32,
    seed: i32,
) -> VoxelMap {
    let builder = ChunkMapBuilder3x1::new(CHUNK_SHAPE, Voxel::EMPTY);
    let mut pyramid = ChunkHashMapPyramid3::new(builder, || SmallKeyHashMap::new(), NUM_LODS);
    let lod0 = pyramid.level_mut(0);

    let chunks = pool.scope(|s| {
        for p in chunks_extent.iter_points() {
            s.spawn(async move {
                let chunk_min = p * CHUNK_SHAPE;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, CHUNK_SHAPE);

                let noise = noise_array(chunk_extent, freq, seed);

                (chunk_min, blocky_voxels_from_noise(&noise, scale))
            });
        }
    });
    for (chunk_key, chunk) in chunks.into_iter() {
        lod0.write_chunk(chunk_key, chunk);
    }

    let index = OctreeChunkIndex::index_chunk_map(SUPERCHUNK_SHAPE, lod0);

    let world_extent = chunks_extent * CHUNK_SHAPE;
    pyramid.downsample_chunks_with_index(&index, &PointDownsampler, &world_extent);

    VoxelMap { pyramid, index }
}

fn blocky_voxels_from_noise(noise: &Array3x1<f32>, scale: f32) -> Array3x1<Voxel> {
    let mut voxels = Array3x1::fill(*noise.extent(), Voxel::EMPTY);

    // Convert the f32 noise into Voxels.
    let sdf_voxel_noise = TransformMap::new(noise, |d: f32| {
        if scale * d < 0.0 {
            Voxel::FILLED
        } else {
            Voxel::EMPTY
        }
    });
    copy_extent(noise.extent(), &sdf_voxel_noise, &mut voxels);

    voxels
}

fn noise_array(extent: Extent3i, freq: f32, seed: i32) -> Array3x1<f32> {
    let min = Point3f::from(extent.minimum);
    let (noise, _min_val, _max_val) = NoiseBuilder::fbm_3d_offset(
        min.x(),
        extent.shape.x() as usize,
        min.y(),
        extent.shape.y() as usize,
        min.z(),
        extent.shape.z() as usize,
    )
    .with_freq(freq)
    .with_seed(seed)
    .generate();

    Array3x1::new_one_channel(extent, noise)
}

pub const CHUNK_LOG2: i32 = 4;
pub const CHUNK_SHAPE: Point3i = PointN([1 << CHUNK_LOG2; 3]);
pub const NUM_LODS: u8 = 6;
pub const SUPERCHUNK_SHAPE: Point3i = PointN([1 << (CHUNK_LOG2 + NUM_LODS as i32 - 1); 3]);
pub const CLIP_BOX_RADIUS: i32 = 8;

pub const WORLD_CHUNKS_EXTENT: Extent3i = Extent3i {
    minimum: PointN([-100, 0, -100]),
    shape: PointN([200, 1, 200]),
};

pub fn world_extent() -> Extent3i {
    WORLD_CHUNKS_EXTENT * CHUNK_SHAPE
}
