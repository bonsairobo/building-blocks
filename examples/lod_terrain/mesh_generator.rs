use crate::voxel_map::VoxelMap;

use bevy_utilities::{
    bevy::{asset::prelude::*, ecs, prelude::*, tasks::ComputeTaskPool},
    mesh::create_mesh_bundle,
    thread_local_resource::ThreadLocalResource,
};
use building_blocks::{
    mesh::*,
    storage::{
        prelude::{ChunkKey3, ClipEvent3},
        SmallKeyHashMap,
    },
};

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
    Create(ChunkKey3),
    Destroy(ChunkKey3),
    Update(ClipEvent3),
}

#[derive(Default)]
pub struct MeshMaterials(pub Vec<Handle<StandardMaterial>>);

#[derive(Default)]
pub struct ChunkMeshes {
    // Map from chunk key to mesh entity.
    entities: SmallKeyHashMap<ChunkKey3, Entity>,
}

/// Generates new meshes for all dirty chunks.
pub fn mesh_generator_system<Map: VoxelMap>(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<Map>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers<Map>>,
    mesh_materials: Res<MeshMaterials>,
    mut mesh_commands: ResMut<MeshCommandQueue>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
    query: Query<&Handle<Mesh>>,
) {
    let new_chunk_meshes = apply_mesh_commands(
        &*voxel_map,
        &*local_mesh_buffers,
        &*pool,
        &mut *mesh_commands,
        &mut *chunk_meshes,
        &mut commands,
        &mut *mesh_assets,
        &query,
    );
    spawn_mesh_entities(
        new_chunk_meshes,
        &*mesh_materials,
        &mut commands,
        &mut *mesh_assets,
        &mut *chunk_meshes,
    );
}

fn apply_mesh_commands<Map: VoxelMap>(
    voxel_map: &Map,
    local_mesh_buffers: &ThreadLocalMeshBuffers<Map>,
    pool: &ComputeTaskPool,
    mesh_commands: &mut MeshCommandQueue,
    chunk_meshes: &mut ChunkMeshes,
    commands: &mut Commands,
    mesh_assets: &mut Assets<Mesh>,
    query: &Query<&Handle<Mesh>>,
) -> Vec<(ChunkKey3, Option<PosNormMesh>)> {
    let max_meshes_per_frame = max_mesh_creations_per_frame(pool);

    let mut num_commands_processed = 0;

    pool.scope(|s| {
        let mut make_mesh = |key: ChunkKey3| {
            s.spawn(async move {
                let mesh_tls = local_mesh_buffers.get();
                let mut mesh_buffers = mesh_tls
                    .get_or_create_with(|| RefCell::new(voxel_map.init_mesh_buffers()))
                    .borrow_mut();

                (key, voxel_map.create_mesh_for_chunk(key, &mut mesh_buffers))
            });
        };

        let mut num_meshes_created = 0;
        for command in mesh_commands.commands.iter().rev().cloned() {
            match command {
                MeshCommand::Create(key) => {
                    num_commands_processed += 1;
                    num_meshes_created += 1;
                    make_mesh(key)
                }
                MeshCommand::Destroy(key) => {
                    num_commands_processed += 1;
                    if let Some(entity) = chunk_meshes.entities.remove(&key) {
                        if let Ok(mesh_handle) = query.get(entity) {
                            mesh_assets.remove(mesh_handle);
                        }
                        commands.entity(entity).despawn();
                    }
                }
                MeshCommand::Update(update) => {
                    num_commands_processed += 1;
                    match update {
                        ClipEvent3::Split(split) => {
                            if let Some(entity) = chunk_meshes.entities.remove(&split.old_chunk) {
                                commands.entity(entity).despawn();
                            }
                            for &key in split.new_chunks.iter() {
                                num_meshes_created += 1;
                                make_mesh(key)
                            }
                        }
                        ClipEvent3::Merge(merge) => {
                            for key in merge.old_chunks.iter() {
                                if let Some(entity) = chunk_meshes.entities.remove(&key) {
                                    commands.entity(entity).despawn();
                                }
                            }
                            num_meshes_created += 1;
                            make_mesh(merge.new_chunk)
                        }
                        ClipEvent3::Enter(key, is_active) => {
                            if is_active {
                                num_meshes_created += 1;
                                make_mesh(key)
                            }
                        }
                        ClipEvent3::Exit(key, was_active) => {
                            if was_active {
                                if let Some(entity) = chunk_meshes.entities.remove(&key) {
                                    commands.entity(entity).despawn();
                                }
                            }
                        }
                    }
                }
            }
            if num_meshes_created >= max_meshes_per_frame {
                break;
            }
        }

        let new_length = mesh_commands.len() - num_commands_processed;
        mesh_commands.commands.truncate(new_length);
    })
}

// ThreadLocal doesn't let you get a mutable reference, so we need to use RefCell. We lock this down to only be used in this
// module as a Local resource, so we know it's safe.
type ThreadLocalMeshBuffers<Map> = ThreadLocalResource<RefCell<<Map as VoxelMap>::MeshBuffers>>;

fn spawn_mesh_entities(
    new_chunk_meshes: Vec<(ChunkKey3, Option<PosNormMesh>)>,
    mesh_materials: &MeshMaterials,
    commands: &mut Commands,
    mesh_assets: &mut Assets<Mesh>,
    chunk_meshes: &mut ChunkMeshes,
) {
    for (chunk_key, item) in new_chunk_meshes.into_iter() {
        let old_mesh = if let Some(mesh) = item {
            chunk_meshes.entities.insert(
                chunk_key,
                commands
                    .spawn_bundle(create_mesh_bundle(
                        mesh,
                        mesh_materials.0[chunk_key.lod as usize].clone(),
                        mesh_assets,
                    ))
                    .id(),
            )
        } else {
            chunk_meshes.entities.remove(&chunk_key)
        };
        if let Some(old_mesh) = old_mesh {
            commands.entity(old_mesh).despawn();
        }
    }
}
