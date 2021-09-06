use crate::VoxelMap;

use bevy_utilities::bevy::{prelude::*, tasks::ComputeTaskPool};

use building_blocks::prelude::{Compression, FastArrayCompressionNx1, FromBytesCompression, Lz4};

const MAX_CACHED_CHUNKS: usize = 100000;
const MAX_CHUNKS_COMPRESSED_PER_FRAME: usize = 200;

/// A system that evicts and compresses the least recently used voxel chunks when the cache gets too big.
pub fn chunk_compression_system(pool: Res<ComputeTaskPool>, mut voxel_map: ResMut<VoxelMap>) {
    voxel_map
        .chunks
        .lod_storage_mut(0)
        .flush_thread_local_caches();

    let num_cached = voxel_map.chunks.lod_storage(0).len_cached();
    if num_cached < MAX_CACHED_CHUNKS {
        return;
    }

    let mut overgrowth = num_cached - MAX_CACHED_CHUNKS;

    let mut num_to_compress = overgrowth.min(MAX_CHUNKS_COMPRESSED_PER_FRAME);

    let lod0_storage = voxel_map.chunks.lod_storage_mut(0);
    let mut nodes_to_compress = Vec::new();
    while nodes_to_compress.len() < num_to_compress {
        if let Some((key, node)) = lod0_storage.remove_lru() {
            if node.user_chunk.is_some() {
                nodes_to_compress.push((key, node));
            } else {
                // TODO: make len_cached only return the number of cached nodes *with data*
                // This node doesn't have any data to compress, so we need to adjust our target.
                overgrowth -= 1;
                num_to_compress = overgrowth.min(MAX_CHUNKS_COMPRESSED_PER_FRAME);
                lod0_storage.insert_node(key, node);
            }
        } else {
            break;
        }
    }

    let compression = FastArrayCompressionNx1::from_bytes_compression(Lz4 { level: 10 });
    let compressed_nodes = pool.scope(|s| {
        for (key, node) in nodes_to_compress.into_iter() {
            s.spawn(async move { (key, node.map(|chunk| compression.compress(&chunk))) });
        }
    });

    let lod0_storage = voxel_map.chunks.lod_storage_mut(0);
    for (key, node) in compressed_nodes.into_iter() {
        let user_chunk = node.user_chunk.unwrap();
        lod0_storage.insert_compressed(key, node.state, user_chunk);
    }
}
