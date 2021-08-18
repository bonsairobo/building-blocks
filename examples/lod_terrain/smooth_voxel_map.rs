use crate::voxel_map::{MapConfig, NoiseConfig, VoxelMap};

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks3};
use building_blocks::{
    mesh::{padded_surface_nets_chunk_extent, surface_nets, PosNormMesh, SurfaceNetsBuffer},
    prelude::*,
    storage::prelude::{ChunkKey3, HashMapChunkTree3x1},
};

const AMBIENT_VALUE: f32 = 1.0;

pub struct SmoothVoxelMap {
    config: MapConfig,
    chunks: HashMapChunkTree3x1<f32>,
}

impl VoxelMap for SmoothVoxelMap {
    type MeshBuffers = MeshBuffers;

    fn generate(pool: &ComputeTaskPool, config: MapConfig) -> Self {
        let MapConfig {
            chunk_exponent,
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

        let chunk_shape = Point3i::fill(1 << chunk_exponent);

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

        let root_lod = num_lods - 1;
        let builder = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
            chunk_shape,
            ambient_value: AMBIENT_VALUE,
            root_lod,
        });
        let mut chunks = builder.build_with_hash_map_storage();

        for (chunk_min, noise) in noise_chunks.into_iter() {
            chunks.write_chunk(ChunkKey::new(0, chunk_min), noise);
        }

        chunks.downsample_extent_into_self(&SdfMeanDownsampler, 0, root_lod, config.world_extent());

        Self { chunks, config }
    }

    fn config(&self) -> &MapConfig {
        &self.config
    }

    fn clipmap_active_chunks(&self, lod0_center: Point3f, active_rx: impl FnMut(ChunkKey3)) {
        self.chunks.clipmap_active_chunks(
            self.config().detail,
            self.config().clip_radius,
            lod0_center,
            |_| true,
            active_rx,
        );
    }

    fn clipmap_events(
        &self,
        old_lod0_center: Point3f,
        new_lod0_center: Point3f,
        update_rx: impl FnMut(ClipEvent3),
    ) {
        self.chunks.clipmap_events(
            self.config().detail,
            self.config().clip_radius,
            old_lod0_center,
            new_lod0_center,
            |_| true,
            update_rx,
        );
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
            true,
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
