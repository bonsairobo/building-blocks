use bevy::tasks::TaskPool;
use building_blocks_core::prelude::*;
use building_blocks_storage::{Array3x1, Array2x1, ChunkUnits, access_traits::*};
use utilities::noise::{noise_array2, noise_array3};

pub fn generate_noise_chunks2(
    pool: &TaskPool,
    chunks_extent: ChunkUnits<Extent2i>,
    chunk_shape: Point2i,
    freq: f32,
    scale: f32,
    seed: i32,
    octaves: u8,
) -> Vec<(Point2i, Array2x1<f32>)> {
    pool.scope(|s| {
        for p in chunks_extent.0.iter_points() {
            s.spawn(async move {
                let chunk_min = p * chunk_shape;
                let chunk_extent = Extent2i::from_min_and_shape(chunk_min, chunk_shape);

                let mut array = noise_array2(chunk_extent, freq, seed, octaves);
                array.for_each_mut(&chunk_extent, |_: (), x| {
                    *x *= scale;
                });

                (chunk_min, array)
            });
        }
    })
}


pub fn generate_noise_chunks3(
    pool: &TaskPool,
    chunks_extent: ChunkUnits<Extent3i>,
    chunk_shape: Point3i,
    freq: f32,
    scale: f32,
    seed: i32,
    octaves: u8,
    subsurface_only: bool,
) -> Vec<(Point3i, Array3x1<f32>)> {
    pool.scope(|s| {
        for p in chunks_extent.0.iter_points() {
            s.spawn(async move {
                let chunk_min = p * chunk_shape;
                let chunk_extent = Extent3i::from_min_and_shape(chunk_min, chunk_shape);

                let mut array = noise_array3(chunk_extent, freq, seed, octaves);

                if subsurface_only {
                    let mut any_negative = false;
                    array.for_each_mut(&chunk_extent, |_: (), x| {
                        *x *= scale;

                        if *x < 0.0 {
                            any_negative = true;
                        }
                    });

                    if any_negative {
                        Some((chunk_min, array))
                    } else {
                        None
                    }
                } else {
                    array.for_each_mut(&chunk_extent, |_: (), x| {
                        *x *= scale;
                    });

                    Some((chunk_min, array))
                }
            });
        }
    }).into_iter().filter_map(|x| x).collect()
}
