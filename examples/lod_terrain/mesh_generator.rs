use crate::{
    voxel_map::{MapConfig, VoxelMap},
    voxel_mesh::VoxelMesh,
    ClipSpheres, SyncBatch,
};

use bevy_utilities::{
    bevy::{
        asset::prelude::*, ecs, prelude::Mesh as BevyMesh, prelude::*, tasks::ComputeTaskPool,
        utils::tracing,
    },
    mesh::create_mesh_bundle,
    thread_local_resource::ThreadLocalResource,
};
use building_blocks::{mesh::*, prelude::*, storage::SmallKeyHashMap};

use std::cell::RefCell;

#[derive(Default)]
pub struct MeshMaterials(pub Vec<Handle<StandardMaterial>>);

#[derive(Default)]
pub struct ChunkMeshes {
    // Map from chunk key to mesh entity.
    entities: SmallKeyHashMap<ChunkKey3, Entity>,
}

/// Generates new meshes for all dirty chunks.
pub fn mesh_generator_system<Mesh: VoxelMesh>(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<VoxelMap>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers<Mesh>>,
    mesh_materials: Res<MeshMaterials>,
    mesh_commands: Res<SyncBatch<LodChange3>>,
    mut mesh_assets: ResMut<Assets<BevyMesh>>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let new_chunk_meshes = apply_mesh_commands::<Mesh>(
        &*voxel_map,
        &*local_mesh_buffers,
        &*pool,
        &*mesh_commands,
        &mut *chunk_meshes,
        &mut commands,
    );

    spawn_mesh_entities(
        new_chunk_meshes,
        &*mesh_materials,
        &mut commands,
        &mut *mesh_assets,
        &mut *chunk_meshes,
    );
}

fn apply_mesh_commands<Mesh: VoxelMesh>(
    voxel_map: &VoxelMap,
    local_mesh_buffers: &ThreadLocalMeshBuffers<Mesh>,
    pool: &ComputeTaskPool,
    mesh_commands: &SyncBatch<LodChange3>,
    chunk_meshes: &mut ChunkMeshes,
    commands: &mut Commands,
) -> Vec<(ChunkKey3, Option<PosNormMesh>)> {
    let make_mesh_span = tracing::info_span!("make_mesh");
    let make_mesh_span_ref = &make_mesh_span;

    let mut chunks_to_remove = Vec::new();

    let new_meshes = pool.scope(|s| {
        let mut make_mesh = |key: ChunkKey3| {
            s.spawn(async move {
                let _trace_guard = make_mesh_span_ref.enter();
                let mesh_tls = local_mesh_buffers.get();
                let mut mesh_buffers = mesh_tls
                    .get_or_create_with(|| {
                        RefCell::new(Mesh::init_mesh_buffers(voxel_map.chunks.chunk_shape()))
                    })
                    .borrow_mut();

                (
                    key,
                    Mesh::create_mesh_for_chunk(&voxel_map.chunks, key, &mut mesh_buffers),
                )
            });
        };

        for command in mesh_commands.take_all().into_iter() {
            match command {
                LodChange3::Split(split) => {
                    chunks_to_remove.push(split.old_chunk);
                    for &key in split.new_chunks.iter() {
                        make_mesh(key);
                    }
                }
                LodChange3::Merge(merge) => {
                    for &key in merge.old_chunks.iter() {
                        chunks_to_remove.push(key);
                    }
                    make_mesh(merge.new_chunk);
                }
            }
        }
    });

    for key in chunks_to_remove.into_iter() {
        if let Some(entity) = chunk_meshes.entities.remove(&key) {
            commands.entity(entity).despawn();
        }
    }

    new_meshes
}

// ThreadLocal doesn't let you get a mutable reference, so we need to use RefCell. We lock this down to only be used in this
// module as a Local resource, so we know it's safe.
type ThreadLocalMeshBuffers<Mesh> = ThreadLocalResource<RefCell<<Mesh as VoxelMesh>::MeshBuffers>>;

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

/// Deletes meshes that aren't bounded by the clip sphere.
pub fn mesh_deleter_system(
    mut commands: Commands,
    config: Res<MapConfig>,
    clip_spheres: Res<ClipSpheres>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let indexer = ChunkIndexer3::new(config.chunk_shape());

    let mut chunks_to_remove = Vec::new();
    for &chunk_key in chunk_meshes.entities.keys() {
        let chunk_sphere = chunk_lod0_bounding_sphere(&indexer, chunk_key);
        if !clip_spheres.new_sphere.intersects(&chunk_sphere) {
            chunks_to_remove.push(chunk_key);
        }
    }

    for chunk_key in chunks_to_remove.into_iter() {
        if let Some(entity) = chunk_meshes.entities.remove(&chunk_key) {
            commands.entity(entity).despawn();
        }
    }
}
