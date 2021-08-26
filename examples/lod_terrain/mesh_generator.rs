use crate::voxel_map::{MapConfig, VoxelMap};

use bevy_utilities::{
    bevy::{
        asset::prelude::*, ecs, prelude::*, render::camera::Camera, tasks::ComputeTaskPool,
        utils::tracing,
    },
    mesh::create_mesh_bundle,
    thread_local_resource::ThreadLocalResource,
};
use building_blocks::{mesh::*, prelude::*, storage::SmallKeyHashMap};

use float_ord::FloatOrd;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

fn max_mesh_creations_per_frame(config: &MapConfig, pool: &ComputeTaskPool) -> usize {
    config.chunks_processed_per_frame_per_core * pool.thread_num()
}

#[derive(Default)]
pub struct MeshCommands {
    /// New commands for this frame.
    new_commands: Mutex<Vec<MeshCommandQueueElem>>,
    is_congested: AtomicBool,
}

impl MeshCommands {
    pub fn add_commands(&self, commands: impl Iterator<Item = MeshCommandQueueElem>) {
        let mut new_commands = self.new_commands.lock().unwrap();
        new_commands.extend(commands);
    }

    pub fn is_congested(&self) -> bool {
        self.is_congested.load(Ordering::Relaxed) || !self.new_commands.lock().unwrap().is_empty()
    }

    pub fn set_congested(&self, is_congested: bool) {
        self.is_congested.store(is_congested, Ordering::Relaxed)
    }
}

#[derive(Default)]
pub struct MeshCommandQueue {
    command_queue: BinaryHeap<MeshCommandQueueElem>,
}

#[derive(Clone, PartialEq)]
pub struct MeshCommandQueueElem {
    dist_to_camera: FloatOrd<f32>,
    command: MeshCommand,
}

impl Eq for MeshCommandQueueElem {}

impl PartialOrd for MeshCommandQueueElem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist_to_camera
            .partial_cmp(&other.dist_to_camera)
            .map(|o| o.reverse())
    }
}

impl Ord for MeshCommandQueueElem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist_to_camera.cmp(&other.dist_to_camera).reverse()
    }
}

impl From<ClipmapSlot3> for MeshCommandQueueElem {
    fn from(slot: ClipmapSlot3) -> Self {
        Self {
            dist_to_camera: FloatOrd(slot.dist),
            command: MeshCommand::Create(slot.key),
        }
    }
}

impl From<LodChange3> for MeshCommandQueueElem {
    fn from(c: LodChange3) -> Self {
        let d = match &c {
            LodChange3::Merge(m) => m.new_chunk_dist,
            LodChange3::Split(s) => s.old_chunk_dist,
        };
        Self {
            dist_to_camera: FloatOrd(d),
            command: MeshCommand::LodChange(c),
        }
    }
}

impl MeshCommandQueue {
    pub fn merge_new_commands(&mut self, new_commands: &MeshCommands) {
        if !self.is_empty() {
            return;
        }

        let new_commands = {
            let mut new_commands = new_commands.new_commands.lock().unwrap();
            std::mem::replace(&mut *new_commands, Vec::new())
        };
        for command in new_commands.into_iter() {
            self.command_queue.push(command);
        }
    }

    pub fn enqueue(&mut self, command: MeshCommandQueueElem) {
        self.command_queue.push(command);
    }

    pub fn is_empty(&self) -> bool {
        self.command_queue.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MeshCommand {
    Create(ChunkKey3),
    LodChange(LodChange3),
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
    config: Res<MapConfig>,
    voxel_map: Res<Map>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers<Map>>,
    mesh_materials: Res<MeshMaterials>,
    new_mesh_commands: Res<MeshCommands>,
    mut mesh_command_queue: ResMut<MeshCommandQueue>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let new_chunk_meshes = apply_mesh_commands(
        &*config,
        &*voxel_map,
        &*local_mesh_buffers,
        &*pool,
        &*new_mesh_commands,
        &mut *mesh_command_queue,
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

fn apply_mesh_commands<Map: VoxelMap>(
    config: &MapConfig,
    voxel_map: &Map,
    local_mesh_buffers: &ThreadLocalMeshBuffers<Map>,
    pool: &ComputeTaskPool,
    new_mesh_commands: &MeshCommands,
    mesh_commands: &mut MeshCommandQueue,
    chunk_meshes: &mut ChunkMeshes,
    commands: &mut Commands,
) -> Vec<(ChunkKey3, Option<PosNormMesh>)> {
    let make_mesh_span = tracing::info_span!("make_mesh");
    let make_mesh_span_ref = &make_mesh_span;

    let max_meshes_per_frame = max_mesh_creations_per_frame(config, pool);

    mesh_commands.merge_new_commands(new_mesh_commands);

    let mut num_commands_processed = 0;

    let new_meshes = pool.scope(|s| {
        let mut make_mesh = |key: ChunkKey3| {
            s.spawn(async move {
                let _trace_guard = make_mesh_span_ref.enter();
                let mesh_tls = local_mesh_buffers.get();
                let mut mesh_buffers = mesh_tls
                    .get_or_create_with(|| RefCell::new(voxel_map.init_mesh_buffers()))
                    .borrow_mut();

                (key, voxel_map.create_mesh_for_chunk(key, &mut mesh_buffers))
            });
        };

        let mut num_meshes_created = 0;
        while let Some(command) = mesh_commands.command_queue.pop() {
            num_commands_processed += 1;
            match command.command {
                MeshCommand::Create(key) => {
                    num_meshes_created += 1;
                    make_mesh(key)
                }
                MeshCommand::LodChange(update) => match update {
                    LodChange3::Split(split) => {
                        if let Some(entity) = chunk_meshes.entities.remove(&split.old_chunk) {
                            commands.entity(entity).despawn();
                        }
                        for &key in split.new_chunks.iter() {
                            num_meshes_created += 1;
                            make_mesh(key)
                        }
                    }
                    LodChange3::Merge(merge) => {
                        for key in merge.old_chunks.iter() {
                            if let Some(entity) = chunk_meshes.entities.remove(&key) {
                                commands.entity(entity).despawn();
                            }
                        }

                        if merge.new_chunk_is_bounded {
                            num_meshes_created += 1;
                            make_mesh(merge.new_chunk)
                        }
                    }
                },
            }
            if num_meshes_created >= max_meshes_per_frame {
                break;
            }
        }
    });

    new_mesh_commands.set_congested(!mesh_commands.command_queue.is_empty());

    new_meshes
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

/// Deletes meshes that aren't bounded by the clip sphere.
pub fn mesh_deleter_system<Map: VoxelMap>(
    mut commands: Commands,
    cameras: Query<(&Camera, &Transform)>,
    voxel_map: Res<Map>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let camera_position = if let Some((_camera, tfm)) = cameras.iter().next() {
        tfm.translation
    } else {
        return;
    };

    let clip_sphere = Sphere3 {
        center: Point3f::from(camera_position),
        radius: voxel_map.config().clip_radius,
    };

    let mut chunks_to_remove = Vec::new();
    for &chunk_key in chunk_meshes.entities.keys() {
        let chunk_sphere = chunk_lod0_bounding_sphere(voxel_map.chunk_indexer(), chunk_key);
        if !clip_sphere.contains(&chunk_sphere) {
            chunks_to_remove.push(chunk_key);
        }
    }

    for chunk_key in chunks_to_remove.into_iter() {
        if let Some(entity) = chunk_meshes.entities.remove(&chunk_key) {
            commands.entity(entity).despawn();
        }
    }
}
