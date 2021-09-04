use crate::{
    voxel_map::{MapConfig, Voxel, VoxelMap},
    ClipSpheres, SyncBatch,
};

use building_blocks::prelude::{Array3x1, ChunkKey3, SdfMeanDownsampler};

use bevy_utilities::bevy::{prelude::*, tasks::ComputeTaskPool, utils::tracing};

/// Inserts new chunks into the `ChunkTree` after they are generated.
pub fn new_chunk_writer_system(
    new_slots: Res<SyncBatch<NewSlot>>,
    mut loaded_chunks: ResMut<LoadedChunks>,
    mut map: ResMut<VoxelMap>,
) {
    let span = tracing::info_span!("chunk_writer");
    let _trace_guard = span.enter();

    for (key, chunk) in loaded_chunks.chunks.drain(..) {
        map.chunks.clipmap_write_loaded_chunk(key, chunk)
    }
    for slot in new_slots.take_all().into_iter() {
        map.chunks.clipmap_mark_node_for_loading(slot.key);
    }
}

pub fn chunk_generator_system(
    config: Res<MapConfig>,
    pool: Res<ComputeTaskPool>,
    mut generate_slots: ResMut<GenerateSlots>,
    mut loaded_chunks: ResMut<LoadedChunks>,
) {
    let span = tracing::info_span!("generate_chunk");
    let span_ref = &span;

    let config_ref = &*config;
    let generated_chunks = pool.scope(|scope| {
        for key in generate_slots.slots.drain(..) {
            scope.spawn(async move {
                let _trace_guard = span_ref.enter();
                (key, VoxelMap::generate_lod0_chunk(config_ref, key.minimum))
            });
        }
    });
    loaded_chunks.chunks.extend(generated_chunks.into_iter());
}

pub fn chunk_downsampler_system(
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<VoxelMap>,
    mut downsample_slots: ResMut<DownsampleSlots>,
    mut loaded_chunks: ResMut<LoadedChunks>,
) {
    let span = tracing::info_span!("chunk_downsampler");
    let _trace_guard = span.enter();

    let chunks_ref = &voxel_map.chunks;
    let generated_chunks = pool.scope(|scope| {
        for dst_chunk_key in downsample_slots.slots.drain(..) {
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
    loaded_chunks.chunks.extend(generated_chunks.into_iter());
}

/// Drives new chunk generation by searching the chunk tree for nodes that need to be loaded.
pub fn find_loading_slots_system(
    clip_spheres: Res<ClipSpheres>,
    voxel_map: Res<VoxelMap>,
    mut generate_slots: ResMut<GenerateSlots>,
    mut downsample_slots: ResMut<DownsampleSlots>,
) {
    let span = tracing::info_span!("find_loading_slots");
    let _trace_guard = span.enter();

    voxel_map.clipmap_loading_slots(clip_spheres.new_sphere, |key| {
        if key.lod == 0 {
            generate_slots.slots.push(key);
        } else {
            downsample_slots.slots.push(key);
        }
    });
}

pub struct NewSlot {
    pub key: ChunkKey3,
}

/// Chunks found to be generated this frame.
#[derive(Default)]
pub struct GenerateSlots {
    slots: Vec<ChunkKey3>,
}

/// Chunks found to be downsampled this frame.
#[derive(Default)]
pub struct DownsampleSlots {
    slots: Vec<ChunkKey3>,
}

/// Chunks loaded this frame.
#[derive(Default)]
pub struct LoadedChunks {
    chunks: Vec<(ChunkKey3, Option<Array3x1<Voxel>>)>,
}
