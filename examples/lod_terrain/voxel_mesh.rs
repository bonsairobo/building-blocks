use crate::voxel_map::Voxel;

use building_blocks::{mesh::*, prelude::*};

pub trait VoxelMesh: 'static + Send {
    type MeshBuffers: Send;

    fn init_mesh_buffers(chunk_shape: Point3i) -> Self::MeshBuffers;

    fn create_mesh_for_chunk(
        chunks: &CompressibleChunkTree3x1<Lz4, Voxel>,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh>;
}

pub struct SmoothMesh;

impl VoxelMesh for SmoothMesh {
    type MeshBuffers = SmoothMeshBuffers;

    fn init_mesh_buffers(chunk_shape: Point3i) -> Self::MeshBuffers {
        let extent = padded_surface_nets_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            chunk_shape,
        ));

        Self::MeshBuffers {
            mesh_buffer: Default::default(),
            neighborhood_buffer: Array3x1::fill(extent, Voxel::EMPTY),
        }
    }

    fn create_mesh_for_chunk(
        chunks: &CompressibleChunkTree3x1<Lz4, Voxel>,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh> {
        if chunks.get_chunk(key).is_none() {
            return None;
        }

        let chunk_extent = chunks.indexer.extent_for_chunk_with_min(key.minimum);
        let padded_chunk_extent = padded_surface_nets_chunk_extent(&chunk_extent);

        // Keep a thread-local cache of buffers to avoid expensive reallocations every time we want to mesh a chunk.
        let Self::MeshBuffers {
            mesh_buffer,
            neighborhood_buffer,
        } = &mut *mesh_buffers;

        // While the chunk shape doesn't change, we need to make sure that it's in the right position for each particular chunk.
        neighborhood_buffer.set_minimum(padded_chunk_extent.minimum);

        copy_extent(
            &padded_chunk_extent,
            &chunks.lod_view(key.lod),
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

pub struct SmoothMeshBuffers {
    mesh_buffer: SurfaceNetsBuffer,
    neighborhood_buffer: Array3x1<Voxel>,
}

pub struct BlockyMesh;

impl VoxelMesh for BlockyMesh {
    type MeshBuffers = BlockyMeshBuffers;

    fn init_mesh_buffers(chunk_shape: Point3i) -> Self::MeshBuffers {
        let extent = padded_greedy_quads_chunk_extent(&Extent3i::from_min_and_shape(
            Point3i::ZERO,
            chunk_shape,
        ));

        Self::MeshBuffers {
            mesh_buffer: GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups()),
            neighborhood_buffer: Array3x1::fill(extent, Voxel::EMPTY),
        }
    }

    fn create_mesh_for_chunk(
        chunks: &CompressibleChunkTree3x1<Lz4, Voxel>,
        key: ChunkKey3,
        mesh_buffers: &mut Self::MeshBuffers,
    ) -> Option<PosNormMesh> {
        if chunks.get_chunk(key).is_none() {
            return None;
        }

        let chunk_extent = chunks.indexer.extent_for_chunk_with_min(key.minimum);
        let padded_chunk_extent = padded_greedy_quads_chunk_extent(&chunk_extent);

        // Keep a thread-local cache of buffers to avoid expensive reallocations every time we want to mesh a chunk.
        let Self::MeshBuffers {
            mesh_buffer,
            neighborhood_buffer,
        } = &mut *mesh_buffers;

        // While the chunk shape doesn't change, we need to make sure that it's in the right position for each particular chunk.
        neighborhood_buffer.set_minimum(padded_chunk_extent.minimum);

        // Only copy the chunk_extent, leaving the padding empty so that we don't get holes on LOD boundaries.
        copy_extent(
            &chunk_extent,
            &chunks.lod_view(key.lod),
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

pub struct BlockyMeshBuffers {
    mesh_buffer: GreedyQuadsBuffer,
    neighborhood_buffer: Array3x1<Voxel>,
}
