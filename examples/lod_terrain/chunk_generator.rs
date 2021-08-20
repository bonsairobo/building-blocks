use crate::voxel_map::VoxelMap;

use bevy_utilities::bevy::{prelude::*, tasks::ComputeTaskPool};
use building_blocks::prelude::{Array3x1, ChunkKey, ChunkKey3, Extent3i, Point3i};

use std::collections::VecDeque;

fn max_chunk_creations_per_frame(pool: &ComputeTaskPool) -> usize {
    40 * pool.thread_num()
}

#[derive(Default)]
pub struct ChunkCommandQueue {
    commands: VecDeque<ChunkCommand>,
}

impl ChunkCommandQueue {
    pub fn enqueue(&mut self, command: ChunkCommand) {
        self.commands.push_front(command);
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChunkCommand {
    Create(ChunkKey3),
    Destroy(ChunkKey3),
    Downsample(ChunkKey3),
}

/// Generates new chunks
pub fn chunk_generator_system<Map: VoxelMap>(
    pool: Res<ComputeTaskPool>,
    mut voxel_map: ResMut<Map>,
    mut chunk_commands: ResMut<ChunkCommandQueue>,
) {
    let (new_chunks, extents_to_downsample, chunks_to_remove) =
        apply_chunk_commands(&*voxel_map, &*pool, &mut *chunk_commands);
    write_chunks(
        &mut *voxel_map,
        new_chunks,
        extents_to_downsample,
        chunks_to_remove,
    );
}

fn apply_chunk_commands<Map: VoxelMap>(
    voxel_map: &Map,
    pool: &ComputeTaskPool,
    chunk_commands: &mut ChunkCommandQueue,
) -> (
    Vec<(Point3i, Option<Array3x1<Map::Voxel>>)>,
    Vec<(Extent3i, u8)>,
    Vec<ChunkKey3>,
) {
    let max_chunks_per_frame = max_chunk_creations_per_frame(pool);

    let mut num_commands_processed = 0;
    let mut chunks_to_remove = Vec::new();
    let mut extents_to_downsample = Vec::new();

    (
        pool.scope(|s| {
            let mut make_chunks = |chunk_min: Point3i| {
                s.spawn(async move { (chunk_min, voxel_map.generate_chunk(chunk_min)) });
            };

            let mut num_chunks_created = 0;
            for command in chunk_commands.commands.iter().rev().cloned() {
                match command {
                    ChunkCommand::Create(key) => {
                        num_commands_processed += 1;
                        num_chunks_created += 1;
                        let chunk_extent = voxel_map.chunk_extent_at_lower_lod(key, 0);
                        if !voxel_map.chunk_is_generated(chunk_extent.minimum, 0) {
                            make_chunks(chunk_extent.minimum)
                        }
                    }
                    ChunkCommand::Destroy(key) => {
                        chunks_to_remove.push(key);
                    }
                    ChunkCommand::Downsample(key) => extents_to_downsample
                        .push((voxel_map.chunk_extent_at_lower_lod(key, 0), key.lod)),
                }
                if num_chunks_created >= max_chunks_per_frame {
                    break;
                }
            }

            let new_length = chunk_commands.len() - num_commands_processed;
            chunk_commands.commands.truncate(new_length);
        }),
        extents_to_downsample,
        chunks_to_remove,
    )
}

fn write_chunks<Map: VoxelMap>(
    voxel_map: &mut Map,
    chunks: Vec<(Point3i, Option<Array3x1<Map::Voxel>>)>,
    extents_to_downsample: Vec<(Extent3i, u8)>,
    chunks_to_remove: Vec<ChunkKey3>,
) {
    for (chunk_min, chunk) in chunks.into_iter() {
        if let Some(chunk) = chunk {
            voxel_map.write_chunk(ChunkKey::new(0, chunk_min), chunk);
        }
    }
    for (extent, max_lod) in extents_to_downsample.into_iter() {
        let mut src_lod = 0;
        for lod in (0..max_lod).rev() {
            src_lod = lod;
            // FIXME: must check all dependent chunks are generated or implicit when this downsample command is processed?
            if voxel_map.chunk_is_generated(extent.minimum, lod) {
                break;
            }
        }
        voxel_map.downsample_extent_into_self(extent, src_lod, max_lod);
    }
    for chunk_key in chunks_to_remove.into_iter() {
        voxel_map.remove_chunk(chunk_key);
    }
}
