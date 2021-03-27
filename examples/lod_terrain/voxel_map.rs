use bevy::tasks::ComputeTaskPool;
use building_blocks::{
    prelude::*,
    storage::{ChunkHashMapPyramid3, OctreeChunkIndex, Sd8, SdfMeanDownsampler, SmallKeyHashMap},
};

use simdnoise::NoiseBuilder;

pub struct VoxelMap {
    pub pyramid: ChunkHashMapPyramid3<Sd8>,
    pub index: OctreeChunkIndex,
}

pub fn generate_map(
    pool: &ComputeTaskPool,
    map_extent: Extent3i,
    freq: f32,
    scale: f32,
    seed: i32,
) -> VoxelMap {
    let builder = ChunkMapBuilder3x1::new(CHUNK_SHAPE, Sd8::ONE);
    let mut pyramid = ChunkHashMapPyramid3::new(builder, || SmallKeyHashMap::new(), NUM_LODS);
    let lod0 = pyramid.level_mut(0);

    let chunks = pool.scope(|s| {
        for p in map_extent.iter_points() {
            s.spawn(async move {
                let chunk_min = p * CHUNK_SHAPE;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, CHUNK_SHAPE);
                let mut chunk_noise = Array3x1::fill(chunk_extent, Sd8::ONE);

                let noise = noise_array(chunk_extent, freq, seed);
                let sdf_voxel_noise = TransformMap::new(&noise, |d: f32| Sd8::from(scale * d));
                copy_extent(&chunk_extent, &sdf_voxel_noise, &mut chunk_noise);

                (chunk_min, chunk_noise)
            });
        }
    });
    for (chunk_key, chunk) in chunks.into_iter() {
        lod0.write_chunk(chunk_key, chunk);
    }

    let index = OctreeChunkIndex::index_chunk_map(SUPERCHUNK_SHAPE, &lod0);

    let world_extent = map_extent * CHUNK_SHAPE;
    pyramid.downsample_chunks_with_index(&index, &SdfMeanDownsampler, &world_extent);

    VoxelMap { pyramid, index }
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
pub const NUM_LODS: u8 = 5;
pub const SUPERCHUNK_SHAPE: Point3i = PointN([1 << (CHUNK_LOG2 + NUM_LODS as i32 - 1); 3]);
pub const CLIP_BOX_RADIUS: i32 = 16;

pub const WORLD_CHUNKS_EXTENT: Extent3i = Extent3i {
    minimum: PointN([-50, 0, -50]),
    shape: PointN([100, 1, 100]),
};
pub const WORLD_EXTENT: Extent3i = Extent3i {
    minimum: PointN([-800, 0, -800]),
    shape: PointN([1600, 16, 1600]),
};
