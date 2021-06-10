use crate::voxel_map::VoxelMap;

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks};
use building_blocks::{
    mesh::{
        greedy_quads, padded_greedy_quads_chunk_extent, GreedyQuadsBuffer, IsOpaque, MergeVoxel,
        PosNormMesh, RIGHT_HANDED_Y_UP_CONFIG,
    },
    prelude::*,
    storage::{ChunkHashMap3x1, ChunkKey3, OctreeChunkIndex},
};

const CHUNK_LOG2: i32 = 4;
const CHUNK_SHAPE: Point3i = PointN([1 << CHUNK_LOG2; 3]);
const NUM_LODS: u8 = 6;
const SUPERCHUNK_SHAPE: Point3i = PointN([1 << (CHUNK_LOG2 + NUM_LODS as i32 - 1); 3]);
const CLIP_BOX_RADIUS: i32 = 8;

const WORLD_CHUNKS_EXTENT: Extent3i = Extent3i {
    minimum: PointN([-100, 0, -100]),
    shape: PointN([200, 1, 200]),
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Voxel(pub u8);

impl Voxel {
    pub const EMPTY: Self = Self(0);
    pub const FILLED: Self = Self(1);
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl MergeVoxel for Voxel {
    type VoxelValue = u8;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.0
    }
}

pub struct BlockyVoxelMap {
    chunks: ChunkHashMap3x1<Voxel>,
    index: OctreeChunkIndex,
}

impl VoxelMap for BlockyVoxelMap {
    type MeshBuffers = MeshBuffers;

    fn generate(pool: &ComputeTaskPool, freq: f32, scale: f32, seed: i32) -> Self {
        let noise_chunks =
            generate_noise_chunks(pool, Self::world_chunks_extent(), CHUNK_SHAPE, freq, seed);

        let builder = ChunkMapBuilder3x1::new(CHUNK_SHAPE, Voxel::EMPTY);
        let mut chunks = builder.build_with_hash_map_storage();

        for (chunk_min, noise) in noise_chunks.into_iter() {
            chunks.write_chunk(
                ChunkKey::new(0, chunk_min),
                blocky_voxels_from_noise(&noise, scale),
            );
        }

        let index = OctreeChunkIndex::index_chunk_map(SUPERCHUNK_SHAPE, &chunks);

        let world_extent = Self::world_chunks_extent() * CHUNK_SHAPE;
        chunks.downsample_chunks_with_index(NUM_LODS, &index, &PointDownsampler, &world_extent);

        Self { chunks, index }
    }

    fn chunk_log2() -> i32 {
        CHUNK_LOG2
    }
    fn clip_box_radius() -> i32 {
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
        let extent = padded_greedy_quads_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            CHUNK_SHAPE,
        ));

        MeshBuffers {
            mesh_buffer: GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups()),
            neighborhood_buffer: Array3x1::fill(extent, Voxel::EMPTY),
        }
    }

    fn create_mesh_for_chunk(
        &self,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh> {
        let chunk_extent = self.chunks.indexer.extent_for_chunk_with_min(key.minimum);
        let padded_chunk_extent = padded_greedy_quads_chunk_extent(&chunk_extent);

        // Keep a thread-local cache of buffers to avoid expensive reallocations every time we want to mesh a chunk.
        let MeshBuffers {
            mesh_buffer,
            neighborhood_buffer,
        } = &mut *mesh_buffers;

        // While the chunk shape doesn't change, we need to make sure that it's in the right position for each particular chunk.
        neighborhood_buffer.set_minimum(padded_chunk_extent.minimum);

        // Only copy the chunk_extent, leaving the padding empty so that we don't get holes on LOD boundaries.
        copy_extent(
            &chunk_extent,
            &self.chunks.lod_view(key.lod),
            neighborhood_buffer,
        );

        let voxel_size = (1 << key.lod) as f32;
        greedy_quads(neighborhood_buffer, &padded_chunk_extent, &mut *mesh_buffer);

        if mesh_buffer.num_quads() == 0 {
            None
        } else {
            let mut mesh = PosNormMesh::default();
            for group in mesh_buffer.quad_groups.iter() {
                for quad in group.quads.iter() {
                    group
                        .face
                        .add_quad_to_pos_norm_mesh(&quad, voxel_size, &mut mesh);
                }
            }

            Some(mesh)
        }
    }
}

fn blocky_voxels_from_noise(noise: &Array3x1<f32>, scale: f32) -> Array3x1<Voxel> {
    let mut voxels = Array3x1::fill(*noise.extent(), Voxel::EMPTY);

    // Convert the f32 noise into Voxels.
    let sdf_voxel_noise = TransformMap::new(noise, |d: f32| {
        if scale * d < 0.0 {
            Voxel::FILLED
        } else {
            Voxel::EMPTY
        }
    });
    copy_extent(noise.extent(), &sdf_voxel_noise, &mut voxels);

    voxels
}

pub struct MeshBuffers {
    mesh_buffer: GreedyQuadsBuffer,
    neighborhood_buffer: Array3x1<Voxel>,
}
