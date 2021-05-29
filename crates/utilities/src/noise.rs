use bevy::tasks::ComputeTaskPool;
use building_blocks_core::prelude::*;
use building_blocks_storage::Array3x1;
use simdnoise::NoiseBuilder;

pub fn generate_noise_chunks(
    pool: &ComputeTaskPool,
    chunks_extent: Extent3i,
    chunk_shape: Point3i,
    freq: f32,
    seed: i32,
) -> Vec<(Point3i, Array3x1<f32>)> {
    pool.scope(|s| {
        for p in chunks_extent.iter_points() {
            s.spawn(async move {
                let chunk_min = p * chunk_shape;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, chunk_shape);

                (chunk_min, noise_array(chunk_extent, freq, seed))
            });
        }
    })
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
