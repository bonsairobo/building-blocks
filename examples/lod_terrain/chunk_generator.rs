use crate::{
    frame_budget::FrameBudget,
    voxel_map::{MapConfig, Voxel, VoxelMap},
    ClipSpheres, SyncBatch,
};

use building_blocks::prelude::{Array3x1, ChunkKey3, SdfMeanDownsampler};

use bevy_utilities::bevy::{
    prelude::*,
    tasks::{ComputeTaskPool, Task},
    utils::tracing,
};

use futures_lite::future;
use std::time::{Duration, Instant};

pub fn chunk_generator_system(
    config: Res<MapConfig>,
    pool: Res<ComputeTaskPool>,
    async_pool: Res<ComputeTaskPool>,
    clip_spheres: Res<ClipSpheres>,
    new_slots: Res<SyncBatch<NewSlot>>,
    mut budget: ResMut<GenerateBudget>,
    mut map: ResMut<VoxelMap>,
    mut generate_tasks: ResMut<GenerateTasks>,
) {
    let mut loaded_chunks = Vec::new();

    // Complete generation tasks.
    budget.0.reset_timer();
    for task in generate_tasks.tasks.drain(..) {
        // PERF: is this the best way to block on many futures?
        let (chunk_key, item, item_duration) = future::block_on(task);
        budget.0.complete_item(item_duration);
        loaded_chunks.push((chunk_key, item));
    }
    budget.0.update_estimate();

    // Mark chunks for loading so we can search for them asynchronously.
    for slot in new_slots.take_all().into_iter() {
        map.chunks.mark_tree_for_loading(slot.key);
    }

    // Find new chunks to load this frame.
    let mut generate_slots = Vec::new();
    let mut downsample_slots = Vec::new();
    {
        let span = tracing::info_span!("find_loading_slots");
        let _trace_guard = span.enter();

        let this_frame_budget = budget.0.request_work(0);

        map.chunks.clipmap_loading_slots(
            this_frame_budget as usize,
            clip_spheres.new_sphere.center,
            |key| {
                if key.lod == 0 {
                    generate_slots.push(key);
                } else {
                    downsample_slots.push(key);
                }
            },
        );
    }

    // Downsample chunks. This is very fast relative to chunk generation.
    {
        let span = tracing::info_span!("downsample_chunks");
        let _trace_guard = span.enter();

        let chunks_ref = &map.chunks;
        let generated_chunks = pool.scope(|scope| {
            for dst_chunk_key in downsample_slots.drain(..) {
                scope.spawn(async move {
                    let mut dst_chunk = chunks_ref.new_ambient_chunk(dst_chunk_key);
                    chunks_ref.downsample_children_into_external(
                        &SdfMeanDownsampler,
                        dst_chunk_key,
                        &mut dst_chunk,
                    );

                    (dst_chunk_key, Some(dst_chunk))
                });
            }
        });
        loaded_chunks.extend(generated_chunks.into_iter());
    }

    // Insert new chunks into the tree.
    {
        let span = tracing::info_span!("write_new_chunks");
        let _trace_guard = span.enter();

        for (key, chunk) in loaded_chunks.drain(..) {
            if let Some(chunk) = chunk {
                map.chunks.write_chunk(key, chunk);
            } else {
                map.chunks.delete_chunk(key);
            }
        }
    }

    // Spawn new chunk generation tasks.
    for key in generate_slots.drain(..) {
        let noise_config = config.noise.clone();
        let chunk_extent = map.chunks.indexer.extent_for_chunk_with_min(key.minimum);
        let task = async_pool.spawn(async move {
            let span = tracing::info_span!("generate_chunk");
            let _trace_guard = span.enter();

            let start_time = Instant::now();
            let chunk = VoxelMap::generate_lod0_chunk(noise_config, chunk_extent);
            (key, chunk, start_time.elapsed())
        });
        generate_tasks.tasks.push(task);
    }
}

pub struct NewSlot {
    pub key: ChunkKey3,
}

pub struct GenerateBudget(pub FrameBudget);

/// All mesh tasks currently running.
#[derive(Default)]
pub struct GenerateTasks {
    tasks: Vec<Task<GenerateTaskOutput>>,
}

pub type GenerateTaskOutput = (ChunkKey3, Option<Array3x1<Voxel>>, Duration);
