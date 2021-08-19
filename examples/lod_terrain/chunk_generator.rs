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
}

/// Generates new chunks
pub fn chunk_generator_system<Map: VoxelMap>(
    pool: Res<ComputeTaskPool>,
    mut voxel_map: ResMut<Map>,
    mut chunk_commands: ResMut<ChunkCommandQueue>,
) {
    let (new_chunks, extents_to_downsample) =
        apply_chunk_commands(&*voxel_map, &*pool, &mut *chunk_commands);
    write_chunks(&mut *voxel_map, new_chunks, extents_to_downsample);
}

fn apply_chunk_commands<Map: VoxelMap>(
    voxel_map: &Map,
    pool: &ComputeTaskPool,
    chunk_commands: &mut ChunkCommandQueue,
) -> (Vec<(Point3i, Option<Array3x1<Map::Voxel>>)>, Vec<Extent3i>) {
    let max_chunks_per_frame = max_chunk_creations_per_frame(pool);

    let mut num_commands_processed = 0;
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
                        let mut needs_downsampling = false;
                        for chunk_min in voxel_map.iter_chunks_for_key(key) {
                            if !voxel_map.chunk_is_generated(chunk_min) {
                                needs_downsampling = true;
                                make_chunks(chunk_min)
                            }
                        }
                        if needs_downsampling {
                            extents_to_downsample.push(voxel_map.chunk_extent_at_lower_lod(key, 0));
                        }
                    }
                }
                if num_chunks_created >= max_chunks_per_frame {
                    break;
                }
            }

            let new_length = chunk_commands.len() - num_commands_processed;
            chunk_commands.commands.truncate(new_length);
        }),
        extents_to_downsample,
    )
}

fn write_chunks<Map: VoxelMap>(
    voxel_map: &mut Map,
    chunks: Vec<(Point3i, Option<Array3x1<Map::Voxel>>)>,
    extents_to_downsample: Vec<Extent3i>,
) {
    for (chunk_min, chunk) in chunks.into_iter() {
        if let Some(chunk) = chunk {
            voxel_map.write_chunk(ChunkKey::new(0, chunk_min), chunk);
        }
    }
    for extent in extents_to_downsample.into_iter() {
        voxel_map.downsample_extent(extent);
    }
}
