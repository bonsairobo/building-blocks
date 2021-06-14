use crate::voxel_map::VoxelMap;

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks};
use building_blocks::{
    mesh::{padded_surface_nets_chunk_extent, surface_nets, PosNormMesh, SurfaceNetsBuffer},
    prelude::*,
    storage::{ChunkHashMap3x1, ChunkKey3, OctreeChunkIndex},
};

const CHUNK_LOG2: i32 = 4;
const CHUNK_SHAPE: Point3i = PointN([1 << CHUNK_LOG2; 3]);
const NUM_LODS: u8 = 4;
const SUPERCHUNK_SHAPE: Point3i = PointN([1 << (CHUNK_LOG2 + NUM_LODS as i32 - 1); 3]);
const CLIP_BOX_RADIUS: u16 = 12;

const WORLD_CHUNKS_EXTENT: Extent3i = Extent3i {
    minimum: PointN([-100, 0, -100]),
    shape: PointN([200, 1, 200]),
};

const AMBIENT_VALUE: f32 = 1.0;

pub struct SmoothVoxelMap {
    chunks: ChunkHashMap3x1<f32>,
    index: OctreeChunkIndex,
}

impl VoxelMap for SmoothVoxelMap {
    type MeshBuffers = MeshBuffers;

    fn generate(pool: &ComputeTaskPool, freq: f32, scale: f32, seed: i32) -> Self {
        let noise_chunks =
            generate_noise_chunks(pool, Self::world_chunks_extent(), CHUNK_SHAPE, freq, seed);

        let builder = ChunkMapBuilder3x1::new(CHUNK_SHAPE, AMBIENT_VALUE);
        let mut chunks = builder.build_with_hash_map_storage();

        for (chunk_min, mut noise) in noise_chunks.into_iter() {
            // Rescale the noise.
            let array = noise.array_mut();
            let extent = *array.extent();
            array.for_each_mut(&extent, |_: (), x| {
                *x *= scale;
            });
            chunks.write_chunk(ChunkKey::new(0, chunk_min), noise);
        }

        let index = OctreeChunkIndex::index_chunk_map(SUPERCHUNK_SHAPE, NUM_LODS, &chunks);

        let world_extent = Self::world_chunks_extent() * CHUNK_SHAPE;
        chunks.downsample_chunks_with_index(&index, &PointDownsampler, &world_extent);

        Self { chunks, index }
    }

    fn chunk_log2() -> i32 {
        CHUNK_LOG2
    }
    fn clip_box_radius() -> u16 {
        CLIP_BOX_RADIUS
    }
    fn world_chunks_extent() -> Extent3i {
        WORLD_CHUNKS_EXTENT
    }
    fn world_extent() -> Extent3i {
        Self::world_chunks_extent() * CHUNK_SHAPE
    }

    fn chunk_index(&self) -> &OctreeChunkIndex {
        &self.index
    }

    fn init_mesh_buffers() -> Self::MeshBuffers {
        let extent = padded_surface_nets_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            CHUNK_SHAPE,
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
