use crate::voxel_map::{MapConfig, NoiseConfig, VoxelMap};

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks3};
use building_blocks::{
    mesh::{padded_surface_nets_chunk_extent, surface_nets, PosNormMesh, SurfaceNetsBuffer},
    prelude::*,
};

const AMBIENT_VALUE: f32 = 1.0;

pub struct SmoothVoxelMap {
    config: MapConfig,
    chunks: HashMapChunkTree3x1<f32>,
}

impl VoxelMap for SmoothVoxelMap {
    type MeshBuffers = MeshBuffers;

    type Chunk = Array3x1<f32>;

    fn generate_lod0(pool: &ComputeTaskPool, config: MapConfig) -> Self {
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

        Self { chunks, config }
    }

    fn downsample_descendants_into_new_chunks(
        &self,
        node_key: ChunkKey3,
        chunk_rx: impl FnMut(ChunkKey3, Self::Chunk),
    ) {
        self.chunks
            .downsample_descendants_into_new_chunks(&PointDownsampler, node_key, 0, chunk_rx)
    }

    fn write_chunk(&mut self, key: ChunkKey3, chunk: Self::Chunk) {
        self.chunks.write_chunk(key, chunk);
    }

    fn config(&self) -> &MapConfig {
        &self.config
    }

    fn clipmap_render_updates(&self, new_clip_sphere: Sphere3, rx: impl FnMut(LodChange3)) {
        self.chunks.clipmap_render_updates(
            self.config().detail,
            new_clip_sphere,
            self.config().chunks_processed_per_frame,
            rx,
        );
    }

    fn chunk_indexer(&self) -> &ChunkIndexer3 {
        &self.chunks.indexer
    }

    fn root_lod(&self) -> u8 {
        self.chunks.root_lod()
    }

    fn visit_root_keys(&self, visitor: impl FnMut(ChunkKey3)) {
        self.chunks.visit_root_keys(visitor);
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
        if self.chunks.get_chunk(key).is_none() {
            return None;
        }

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
