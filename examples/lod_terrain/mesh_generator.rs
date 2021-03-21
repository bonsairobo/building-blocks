use crate::voxel_map::VoxelMap;

use building_blocks::{
    mesh::*,
    prelude::*,
    storage::{LodChunkKey3, LodChunkUpdate3, Sd8, SmallKeyHashMap},
};
use utilities::bevy_util::{mesh::create_mesh_bundle, thread_local_resource::ThreadLocalResource};

use bevy::{asset::prelude::*, ecs, prelude::*, tasks::ComputeTaskPool};
use std::{cell::RefCell, collections::VecDeque};

fn max_mesh_creations_per_frame(pool: &ComputeTaskPool) -> usize {
    40 * pool.thread_num()
}

#[derive(Default)]
pub struct MeshCommandQueue {
    commands: VecDeque<MeshCommand>,
}

impl MeshCommandQueue {
    pub fn enqueue(&mut self, command: MeshCommand) {
        self.commands.push_front(command);
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }
}

// PERF: try to eliminate the use of multiple Vecs
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshCommand {
    Create(LodChunkKey3),
    Update(LodChunkUpdate3),
}

#[derive(Default)]
pub struct MeshMaterial(pub Handle<StandardMaterial>);

#[derive(Default)]
pub struct ChunkMeshes {
    // Map from chunk key to mesh entity.
    entities: SmallKeyHashMap<LodChunkKey3, Entity>,
}

/// Generates new meshes for all dirty chunks.
pub fn mesh_generator_system(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<VoxelMap>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers>,
    mesh_material: Res<MeshMaterial>,
    mut mesh_commands: ResMut<MeshCommandQueue>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let new_chunk_meshes = apply_mesh_commands(
        &*voxel_map,
        &*local_mesh_buffers,
        &*pool,
        &mut *mesh_commands,
        &mut *chunk_meshes,
        &mut commands,
    );
    spawn_mesh_entities(
        new_chunk_meshes,
        &*mesh_material,
        &mut commands,
        &mut *mesh_assets,
        &mut *chunk_meshes,
    );
}

fn apply_mesh_commands(
    voxel_map: &VoxelMap,
    local_mesh_buffers: &ThreadLocalMeshBuffers,
    pool: &ComputeTaskPool,
    mesh_commands: &mut MeshCommandQueue,
    chunk_meshes: &mut ChunkMeshes,
    commands: &mut Commands,
) -> Vec<(LodChunkKey3, Option<PosNormMesh>)> {
    let num_chunks_to_mesh = mesh_commands.len().min(max_mesh_creations_per_frame(pool));

    let mut num_creates = 0;
    let mut num_updates = 0;
    pool.scope(|s| {
        let mut num_meshes_created = 0;
        for command in mesh_commands.commands.iter().rev().cloned() {
            match command {
                MeshCommand::Create(key) => {
                    num_creates += 1;
                    num_meshes_created += 1;
                    s.spawn(async move {
                        (
                            key,
                            create_mesh_for_chunk(key, voxel_map, local_mesh_buffers),
                        )
                    });
                }
                MeshCommand::Update(update) => {
                    num_updates += 1;
                    match update {
                        LodChunkUpdate3::Split(split) => {
                            if let Some(entity) = chunk_meshes.entities.remove(&split.old_chunk) {
                                commands.despawn(entity);
                            }
                            for &key in split.new_chunks.iter() {
                                num_meshes_created += 1;
                                s.spawn(async move {
                                    (
                                        key,
                                        create_mesh_for_chunk(key, voxel_map, local_mesh_buffers),
                                    )
                                });
                            }
                        }
                        LodChunkUpdate3::Merge(merge) => {
                            for key in merge.old_chunks.iter() {
                                if let Some(entity) = chunk_meshes.entities.remove(&key) {
                                    commands.despawn(entity);
                                }
                            }
                            num_meshes_created += 1;
                            s.spawn(async move {
                                (
                                    merge.new_chunk,
                                    create_mesh_for_chunk(
                                        merge.new_chunk,
                                        voxel_map,
                                        local_mesh_buffers,
                                    ),
                                )
                            });
                        }
                    }
                }
            }
            if num_meshes_created >= num_chunks_to_mesh {
                break;
            }
        }

        let new_length = mesh_commands.len() - (num_creates + num_updates);
        mesh_commands.commands.truncate(new_length);
    })
}

fn create_mesh_for_chunk(
    key: LodChunkKey3,
    voxel_map: &VoxelMap,
    local_mesh_buffers: &ThreadLocalMeshBuffers,
) -> Option<PosNormMesh> {
    let chunks = voxel_map.pyramid.level(key.lod);

    let padded_chunk_extent =
        padded_surface_nets_chunk_extent(&chunks.indexer.extent_for_chunk_at_key(key.chunk_key));
    let mut padded_chunk = Array3x1::fill(padded_chunk_extent, Sd8::ONE);
    copy_extent(&padded_chunk_extent, chunks, &mut padded_chunk);

    let mesh_tls = local_mesh_buffers.get();
    let mut surface_nets_buffer = mesh_tls.get_or_default().borrow_mut();
    let voxel_size = (1 << key.lod) as f32;
    surface_nets(
        &padded_chunk,
        &padded_chunk_extent,
        voxel_size,
        &mut *surface_nets_buffer,
    );

    if surface_nets_buffer.mesh.indices.is_empty() {
        None
    } else {
        Some(surface_nets_buffer.mesh.clone())
    }
}

// ThreadLocal doesn't let you get a mutable reference, so we need to use RefCell. We lock this down to only be used in this
// module as a Local resource, so we know it's safe.
type ThreadLocalMeshBuffers = ThreadLocalResource<RefCell<SurfaceNetsBuffer>>;

fn spawn_mesh_entities(
    new_chunk_meshes: Vec<(LodChunkKey3, Option<PosNormMesh>)>,
    mesh_material: &MeshMaterial,
    commands: &mut Commands,
    mesh_assets: &mut Assets<Mesh>,
    chunk_meshes: &mut ChunkMeshes,
) {
    for (lod_chunk_key, item) in new_chunk_meshes.into_iter() {
        let old_mesh = if let Some(mesh) = item {
            chunk_meshes.entities.insert(
                lod_chunk_key,
                commands
                    .spawn(create_mesh_bundle(
                        mesh,
                        mesh_material.0.clone(),
                        mesh_assets,
                    ))
                    .current_entity()
                    .unwrap(),
            )
        } else {
            chunk_meshes.entities.remove(&lod_chunk_key)
        };
        if let Some(old_mesh) = old_mesh {
            commands.despawn(old_mesh);
        }
    }
}
