use crate::VoxelMap;

use bevy_utilities::{
    bevy::{prelude::*, tasks::ComputeTaskPool},
};

use building_blocks::prelude::{Compression, FromBytesCompression, Lz4, FastArrayCompressionNx1};

const MAX_CACHED_CHUNKS: usize = 100000;
const MAX_CHUNKS_COMPRESSED_PER_FRAME: usize = 200;

/// A system that evicts and compresses the least recently used voxel chunks when the cache gets too big.
pub fn chunk_compression_system(
    pool: Res<ComputeTaskPool>,
    mut voxel_map: ResMut<VoxelMap>,
) {
    voxel_map.chunks.lod_storage_mut(0).flush_thread_local_caches();

    let num_cached = voxel_map.chunks.lod_storage(0).len_cached();
    if num_cached < MAX_CACHED_CHUNKS {
        return;
    }

    let overgrowth = num_cached - MAX_CACHED_CHUNKS;

    let num_to_compress = overgrowth.min(MAX_CHUNKS_COMPRESSED_PER_FRAME);

    let mut nodes_to_compress = Vec::new();
    for _ in 0..num_to_compress {
        if let Some(key_and_node) = voxel_map.chunks.lod_storage_mut(0).remove_lru() {
            nodes_to_compress.push(key_and_node);
        } else {
            break;
        }
    }

    let compression = FastArrayCompressionNx1::from_bytes_compression(Lz4 { level: 10 });
    let compressed_nodes = pool.scope(|s| {
        for (key, node) in nodes_to_compress.into_iter() {
            s.spawn(async move {
                (key, node.map(|chunk| compression.compress(&chunk)))
            });
        }
    });

    for (key, compressed_node) in compressed_nodes.into_iter() {
        voxel_map
            .chunks
            .lod_storage_mut(0)
            .insert_compressed(key, compressed_node);
    }
}
