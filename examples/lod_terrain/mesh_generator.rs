use crate::{
    frame_budget::FrameBudget,
    voxel_map::{MapConfig, VoxelMap},
    voxel_mesh::VoxelMesh,
    ClipSpheres,
};

use bevy_utilities::{
    bevy::{
        asset::prelude::*,
        ecs,
        prelude::Mesh as BevyMesh,
        prelude::*,
        tasks::{ComputeTaskPool, Task},
        utils::tracing,
    },
    mesh::create_mesh_bundle,
    thread_local_resource::ThreadLocalResource,
};
use building_blocks::{mesh::*, prelude::*, storage::SmallKeyHashMap};

use futures_lite::future;
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Starts and polls tasks to generate new meshes.
///
/// In order to have tasks running for the full duration of a frame, we first poll all outstanding tasks to completion, then
/// spawn new ones.
pub fn mesh_generator_system<Mesh: VoxelMesh>(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<VoxelMap>,
    clip_spheres: Res<ClipSpheres>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers<Mesh>>,
    mesh_materials: Res<MeshMaterials>,
    mut mesh_assets: ResMut<Assets<BevyMesh>>,
    mut budget: ResMut<MeshBudget>,
    mut mesh_tasks: ResMut<MeshTasks>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    // Atomically spawn and remove mesh entities for this frame.
    spawn_and_despawn_mesh_entities(
        &mut *mesh_tasks,
        &mut budget.0,
        &*mesh_materials,
        &mut commands,
        &mut *mesh_assets,
        &mut *chunk_meshes,
    );

    // Find render updates.
    let mut updates = Vec::new();
    let span = tracing::info_span!("lod_changes");
    {
        let _trace_guard = span.enter();

        let this_frame_budget = budget.0.request_work(0);

        voxel_map.chunks.clipmap_render_updates(
            voxel_map.config.detail,
            clip_spheres.new_sphere.center,
            this_frame_budget as usize,
            |c| updates.push(c),
        );
    }

    start_mesh_tasks::<Mesh>(
        &*voxel_map,
        &*local_mesh_buffers,
        &*pool,
        updates,
        &mut *mesh_tasks,
    );
}

fn start_mesh_tasks<Mesh: VoxelMesh>(
    voxel_map: &VoxelMap,
    local_mesh_buffers: &ThreadLocalMeshBuffers<Mesh>,
    pool: &ComputeTaskPool,
    updates: Vec<LodChange3>,
    mesh_tasks: &mut MeshTasks,
) {
    let MeshTasks { tasks, removals } = mesh_tasks;

    let mut start_task = |key: ChunkKey3| {
        if voxel_map.chunks.get_chunk(key).is_none() {
            return;
        }

        let chunk_shape = voxel_map.chunks.chunk_shape();
        let task_local_mesh_buffers = local_mesh_buffers.get();
        let neighborhood = Mesh::copy_chunk_neighborhood(&voxel_map.chunks, key);

        let task = pool.spawn(async move {
            let span = tracing::info_span!("make_mesh");
            let _trace_guard = span.enter();

            let start_time = Instant::now();

            let mut mesh_buffers = task_local_mesh_buffers
                .get_or_create_with(|| RefCell::new(Mesh::init_mesh_buffers(chunk_shape)))
                .borrow_mut();

            let mesh = Mesh::create_mesh_for_chunk(key, &neighborhood, &mut mesh_buffers);

            (key, mesh, start_time.elapsed())
        });
        tasks.push(task);
    };

    for update in updates.into_iter() {
        match update {
            LodChange3::Spawn(key) => {
                start_task(key);
            }
            LodChange3::Split(split) => {
                removals.push(split.old_chunk);
                for &key in split.new_chunks.iter() {
                    start_task(key);
                }
            }
            LodChange3::Merge(merge) => {
                for &key in merge.old_chunks.iter() {
                    removals.push(key);
                }
                start_task(merge.new_chunk);
            }
        }
    }
}

// ThreadLocal doesn't let you get a mutable reference, so we need to use RefCell. We lock this down to only be used in this
// module as a Local resource, so we know it's safe.
type ThreadLocalMeshBuffers<Mesh> = ThreadLocalResource<RefCell<<Mesh as VoxelMesh>::MeshBuffers>>;

fn spawn_and_despawn_mesh_entities(
    mesh_tasks: &mut MeshTasks,
    budget: &mut FrameBudget,
    mesh_materials: &MeshMaterials,
    commands: &mut Commands,
    mesh_assets: &mut Assets<Mesh>,
    chunk_meshes: &mut ChunkMeshes,
) {
    budget.reset_timer();

    // Finish all outstanding tasks.
    for task in mesh_tasks.tasks.drain(..) {
        // PERF: is this the best way to block on many futures?
        let (chunk_key, item, item_duration) = future::block_on(task);

        budget.complete_item(item_duration);

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

    budget.update_estimate();

    for chunk_key in mesh_tasks.removals.drain(..) {
        if let Some(mesh_entity) = chunk_meshes.entities.remove(&chunk_key) {
            commands.entity(mesh_entity).despawn();
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

#[derive(Default)]
pub struct MeshMaterials(pub Vec<Handle<StandardMaterial>>);

#[derive(Default)]
pub struct ChunkMeshes {
    // Map from chunk key to mesh entity.
    entities: SmallKeyHashMap<ChunkKey3, Entity>,
}

pub struct MeshBudget(pub FrameBudget);

/// All mesh tasks currently running.
#[derive(Default)]
pub struct MeshTasks {
    tasks: Vec<Task<MeshTaskOutput>>,
    // These need to be applied in the same frame when the new meshes are created so splits/merges happen atomically.
    removals: Vec<ChunkKey3>,
}

pub type MeshTaskOutput = (ChunkKey3, Option<PosNormMesh>, Duration);
