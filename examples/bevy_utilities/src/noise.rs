use bevy::tasks::TaskPool;
use building_blocks_core::prelude::*;
use building_blocks_storage::{Array3x1, Array3x3};
use utilities::noise::{noise_array, noise_array_warp};

pub fn generate_noise_chunks(
    pool: &TaskPool,
    chunks_extent: Extent3i,
    chunk_shape: Point3i,
    freq: f32,
    seed: i32,
    octaves: u8,
) -> Vec<(Point3i, Array3x1<f32>)> {
    pool.scope(|s| {
        for p in chunks_extent.iter_points() {
            s.spawn(async move {
                let chunk_min = p * chunk_shape;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, chunk_shape);

                (chunk_min, noise_array(chunk_extent, freq, seed, octaves))
            });
        }
    })
}

pub fn generate_noise_chunks_warp(
    pool: &TaskPool,
    chunks_extent: Extent3i,
    chunk_shape: Point3i,
    freq: f32,
    seed: i32,
) -> Vec<(Point3i, Array3x3<f32, f32, f32>)> {
    pool.scope(|s| {
        for p in chunks_extent.iter_points() {
            s.spawn(async move {
                let chunk_min = p * chunk_shape;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, chunk_shape);

                (chunk_min, noise_array_warp(chunk_extent, freq, seed))
            });
        }
    })
}
