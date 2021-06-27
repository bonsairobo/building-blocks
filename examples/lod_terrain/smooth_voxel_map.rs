use crate::voxel_map::{MapConfig, NoiseConfig, VoxelMap};

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks3};
use building_blocks::{
    mesh::{padded_surface_nets_chunk_extent, surface_nets, PosNormMesh, SurfaceNetsBuffer},
    prelude::*,
    storage::{ChunkHashMap3x1, ChunkKey3, OctreeChunkIndex},
};

const AMBIENT_VALUE: f32 = 1.0;

pub struct SmoothVoxelMap {
    config: MapConfig,
    chunks: ChunkHashMap3x1<f32>,
    index: OctreeChunkIndex,
}

impl VoxelMap for SmoothVoxelMap {
    type MeshBuffers = MeshBuffers;

    fn generate(pool: &ComputeTaskPool, config: MapConfig) -> Self {
        let MapConfig {
            superchunk_shape,
            chunk_shape,
            num_lods,
            world_chunks_extent,
            noise:
                NoiseConfig {
                    freq,
                    scale,
                    seed,
                    octaves,
                },
            ..
        } = config;

        let noise_chunks = generate_noise_chunks3(
            pool,
            world_chunks_extent,
            chunk_shape,
            freq,
            scale,
            seed,
            octaves,
            true,
        );

        let builder = ChunkMapBuilder3x1::new(chunk_shape, AMBIENT_VALUE);
        let mut chunks = builder.build_with_hash_map_storage();

        for (chunk_min, noise) in noise_chunks.into_iter() {
            chunks.write_chunk(ChunkKey::new(0, chunk_min), noise);
        }

        let index = OctreeChunkIndex::index_chunk_map(superchunk_shape, num_lods, &chunks);

        chunks.downsample_chunks_with_index(&index, &PointDownsampler, &config.world_extent());

        Self {
            chunks,
            index,
            config,
        }
    }

    fn config(&self) -> &MapConfig {
        &self.config
    }

    fn chunk_index(&self) -> &OctreeChunkIndex {
        &self.index
    }

    fn init_mesh_buffers(&self) -> Self::MeshBuffers {
        let extent = padded_surface_nets_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            self.chunks.chunk_shape(),
        ));

        MeshBuffers {
            mesh_buffer: Default::default(),
            neighborhood_buffer: Array3x1::fill(extent, AMBIENT_VALUE),
        }
    }

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh> {
        let chunk_extent = self.chunks.indexer.extent_for_chunk_with_min(key.minimum);
        let padded_chunk_extent = padded_surface_nets_chunk_extent(&chunk_extent);

        // Keep a thread-local cache of buffers to avoid expensive reallocations every time we want to mesh a chunk.
        let MeshBuffers {
            mesh_buffer,
            neighborhood_buffer,
        } = &mut *mesh_buffers;

        // While the chunk shape doesn't change, we need to make sure that it's in the right position for each particular chunk.
        neighborhood_buffer.set_minimum(padded_chunk_extent.minimum);

        copy_extent(
            &padded_chunk_extent,
            &self.chunks.lod_view(key.lod),
            neighborhood_buffer,
        );

        let voxel_size = (1 << key.lod) as f32;
        surface_nets(
            neighborhood_buffer,
            &padded_chunk_extent,
            voxel_size,
            &mut *mesh_buffer,
        );

        if mesh_buffer.mesh.indices.is_empty() {
            None
        } else {
            Some(mesh_buffer.mesh.clone())
        }
    }
}

pub struct MeshBuffers {
    mesh_buffer: SurfaceNetsBuffer,
    neighborhood_buffer: Array3x1<f32>,
}
