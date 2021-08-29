use crate::voxel_map::{MapConfig, NoiseConfig, VoxelMap};

use bevy_utilities::{bevy::tasks::ComputeTaskPool, noise::generate_noise_chunks3};
use building_blocks::{
    mesh::{
        greedy_quads, padded_greedy_quads_chunk_extent, GreedyQuadsBuffer, IsOpaque, MergeVoxel,
        PosNormMesh, RIGHT_HANDED_Y_UP_CONFIG,
    },
    prelude::*,
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
    chunks: HashMapChunkTree3x1<Voxel>,
    config: MapConfig,
}

impl VoxelMap for BlockyVoxelMap {
    type MeshBuffers = MeshBuffers;

    type Chunk = Array3x1<Voxel>;

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
            ambient_value: Voxel::EMPTY,
            root_lod,
        });
        let mut chunks = builder.build_with_hash_map_storage();

        for (chunk_min, noise) in noise_chunks.into_iter() {
            chunks.write_chunk(ChunkKey::new(0, chunk_min), blocky_voxels_from_sdf(&noise));
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

    fn clipmap_new_chunks(
        &self,
        old_clip_sphere: Sphere3,
        new_clip_sphere: Sphere3,
        rx: impl FnMut(ClipmapSlot3),
    ) {
        self.chunks.clipmap_new_chunks(
            self.config().detail,
            self.config().min_enter_lod,
            old_clip_sphere,
            new_clip_sphere,
            rx,
        );
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
        let extent = padded_greedy_quads_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            self.config.chunk_shape(),
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
        if self.chunks.get_chunk(key).is_none() {
            return None;
        }

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

fn blocky_voxels_from_sdf(sdf: &Array3x1<f32>) -> Array3x1<Voxel> {
    let mut voxels = Array3x1::fill(*sdf.extent(), Voxel::EMPTY);

    // Convert the f32 SDF into Voxels.
    let sdf_voxel_noise =
        TransformMap::new(
            sdf,
            |d: f32| {
                if d < 0.0 {
                    Voxel::FILLED
                } else {
                    Voxel::EMPTY
                }
            },
        );
    copy_extent(sdf.extent(), &sdf_voxel_noise, &mut voxels);

    voxels
}

pub struct MeshBuffers {
    mesh_buffer: GreedyQuadsBuffer,
    neighborhood_buffer: Array3x1<Voxel>,
}
