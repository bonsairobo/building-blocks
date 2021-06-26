use building_blocks_core::prelude::*;
use building_blocks_storage::Array3x1;
use simdnoise::NoiseBuilder;

pub fn noise_array(extent: Extent3i, freq: f32, seed: i32, octaves: u8) -> Array3x1<f32> {
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
    .with_octaves(octaves)
    .generate();

    Array3x1::new_one_channel(extent, noise)
}
