use building_blocks::core::{
    prelude::*,
    sdfu::{self, SDF},
};
use building_blocks::mesh::*;
use building_blocks::storage::prelude::*;

use bevy_utilities::{
    bevy::{
        prelude::*,
        render::wireframe::WireframeConfig,
        tasks::{ComputeTaskPool, TaskPool},
    },
    mesh::create_mesh_bundle,
    noise::generate_noise_chunks3,
};

pub struct MeshGeneratorState {
    current_shape_index: i32,
    flat_shaded: bool,
    chunk_mesh_entities: Vec<Entity>,
}

impl MeshGeneratorState {
    pub fn new() -> Self {
        Self {
            current_shape_index: 0,
            flat_shaded: false,
            chunk_mesh_entities: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Sdf(Sdf),
    SdfNoise,
    HeightMap(HeightMap),
    Blocky(Blocky),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sdf {
    Cube,
    Sphere,
    Torus,
}

impl Sdf {
    fn get_sdf(&self) -> Box<dyn Fn(Point3i) -> Sd16> {
        match self {
            Sdf::Cube => {
                let cube = sdfu::Box::new(Point3f::fill(20.0));

                Box::new(move |p| Sd16::from(cube.dist(Point3f::from(p))))
            }
            Sdf::Sphere => {
                let sphere = sdfu::Sphere::new(20.0);

                Box::new(move |p| Sd16::from(sphere.dist(Point3f::from(p))))
            }
            Sdf::Torus => {
                let torus = sdfu::Torus::new(4.0, 16.0);

                Box::new(move |p| Sd16::from(torus.dist(Point3f::from(p.yzx()))))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeightMap {
    Wave,
}

impl HeightMap {
    fn get_height_map(&self) -> impl Fn(Point2i) -> f32 {
        match self {
            HeightMap::Wave => {
                |p: Point2i| 10.0 * (1.0 + (0.2 * p.x() as f32).cos() + (0.2 * p.y() as f32).sin())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Blocky {
    Terrace,
}

impl Blocky {
    fn get_voxels(&self) -> Array3x1<CubeVoxel> {
        match self {
            Blocky::Terrace => {
                let extent =
                    Extent3i::from_min_and_shape(Point3i::fill(-20), Point3i::fill(40)).padded(1);
                let mut voxels = Array3x1::fill(extent, CubeVoxel(false));
                for i in 0..40 {
                    let level = Extent3i::from_min_and_shape(
                        Point3i::fill(i - 20),
                        PointN([40 - i, 1, 40 - i]),
                    );
                    voxels.fill_extent(&level, CubeVoxel(true));
                }

                voxels
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CubeVoxel(bool);

#[derive(Eq, PartialEq)]
struct TrivialMergeValue;

impl MergeVoxel for CubeVoxel {
    type VoxelValue = TrivialMergeValue;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        TrivialMergeValue
    }
}

impl IsOpaque for CubeVoxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl IsEmpty for CubeVoxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}

const NUM_SHAPES: i32 = 6;

fn choose_shape(index: i32) -> Shape {
    match index {
        0 => Shape::Sdf(Sdf::Cube),
        1 => Shape::Sdf(Sdf::Sphere),
        2 => Shape::Sdf(Sdf::Torus),
        3 => Shape::SdfNoise,
        4 => Shape::HeightMap(HeightMap::Wave),
        5 => Shape::Blocky(Blocky::Terrace),
        _ => panic!("bad shape index"),
    }
}

#[derive(Default)]
pub struct MeshMaterial(pub Handle<StandardMaterial>);

pub fn mesh_generator_system(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    mut wireframe_config: ResMut<WireframeConfig>,
    mut state: ResMut<MeshGeneratorState>,
    mut meshes: ResMut<Assets<Mesh>>,
    keyboard_input: Res<Input<KeyCode>>,
    material: Res<MeshMaterial>,
) {
    let mut new_shape_requested = false;
    if keyboard_input.just_pressed(KeyCode::Left) {
        new_shape_requested = true;
        state.current_shape_index = (state.current_shape_index - 1).rem_euclid(NUM_SHAPES);
    } else if keyboard_input.just_pressed(KeyCode::Right) {
        new_shape_requested = true;
        state.current_shape_index = (state.current_shape_index + 1).rem_euclid(NUM_SHAPES);
    }
    if keyboard_input.just_pressed(KeyCode::F) {
        new_shape_requested = true;
        state.flat_shaded = !state.flat_shaded;
    }
    if keyboard_input.just_pressed(KeyCode::W) {
        wireframe_config.global = !wireframe_config.global;
    }

    if new_shape_requested || state.chunk_mesh_entities.is_empty() {
        // Delete the old meshes.
        for entity in state.chunk_mesh_entities.drain(..) {
            commands.entity(entity).despawn();
        }

        // Sample the new shape.
        let chunk_meshes = match choose_shape(state.current_shape_index) {
            Shape::Sdf(sdf) => generate_chunk_meshes_from_sdf(sdf, &pool.0, state.flat_shaded),
            Shape::SdfNoise => generate_chunk_meshes_from_sdf_noise(&pool.0, state.flat_shaded),
            Shape::HeightMap(hm) => generate_chunk_meshes_from_height_map(hm, &pool.0, state.flat_shaded),
            Shape::Blocky(blocky) => generate_chunk_meshes_from_blocky(blocky, &pool.0),
        };

        for mesh in chunk_meshes.into_iter() {
            if let Some(mesh) = mesh {
                let entity = commands
                    .spawn_bundle(create_mesh_bundle(mesh, material.0.clone(), &mut meshes))
                    .id();
                state.chunk_mesh_entities.push(entity);
            }
        }
    }
}

fn generate_chunk_meshes_from_sdf(
    sdf: Sdf,
    pool: &TaskPool,
    flat_shaded: bool,
) -> Vec<Option<PosNormMesh>> {
    let sdf = sdf.get_sdf();
    let sample_extent =
        Extent3i::from_min_and_shape(Point3i::fill(-20), Point3i::fill(40)).padded(1);

    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3x1 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let builder = ChunkMapBuilder3x1::new(ChunkMapConfig {
        chunk_shape: Point3i::fill(16),
        ambient_value: Sd16::ONE,
        root_lod: 0,
    });
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(&sample_extent, &Func(sdf), &mut map.lod_view_mut(0));

    generate_surface_nets_meshes(pool, &map, flat_shaded)
}

fn generate_chunk_meshes_from_sdf_noise(
    pool: &TaskPool,
    flat_shaded: bool,
) -> Vec<Option<PosNormMesh>> {
    let chunks_extent = ChunkUnits(Extent3i::from_min_and_shape(
        PointN::fill(-1),
        PointN::fill(2),
    ));
    let freq = 0.15;
    let seed = 313;
    let noise_chunks = generate_noise_chunks3(
        pool,
        chunks_extent,
        PointN::fill(16),
        freq,
        1.0,
        seed,
        3,
        true,
    );

    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3x1 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let builder = ChunkMapBuilder3x1::new(ChunkMapConfig { chunk_shape: PointN([16; 3]), ambient_value: 99999.0, root_lod: 0 });
    let mut map = builder.build_with_hash_map_storage();
    for (chunk_min, chunk) in noise_chunks.into_iter() {
        map.write_chunk(ChunkKey::new(0, chunk_min), chunk);
    }

    generate_surface_nets_meshes(pool, &map, flat_shaded)
}

fn generate_surface_nets_meshes<T: 'static + Clone + Send + Sync + SignedDistance>(
    pool: &TaskPool,
    map: &ChunkHashMap3x1<T>,
    flat_shaded: bool,
) -> Vec<Option<PosNormMesh>> {
    pool.scope(|s| {
        for chunk_key in map.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_surface_nets_chunk_extent(
                    &map.indexer.extent_for_chunk_with_min(chunk_key.minimum),
                );
                let mut padded_chunk = Array3x1::fill(padded_chunk_extent, map.ambient_value());
                copy_extent(&padded_chunk_extent, &map.lod_view(0), &mut padded_chunk);

                let mut surface_nets_buffer = SurfaceNetsBuffer::default();
                let voxel_size = 1.0;
                surface_nets(
                    &padded_chunk,
                    &padded_chunk_extent,
                    voxel_size,
                    !flat_shaded,
                    &mut surface_nets_buffer,
                );
                if flat_shaded {
                    surface_nets_buffer.mesh = surface_nets_buffer.mesh.process_for_flat_shading();
                }

                if surface_nets_buffer.mesh.positions.is_empty() {
                    None
                } else {
                    Some(surface_nets_buffer.mesh)
                }
            })
        }
    })
}

fn generate_chunk_meshes_from_height_map(
    hm: HeightMap,
    pool: &TaskPool,
    flat_shaded: bool
) -> Vec<Option<PosNormMesh>> {
    let height_map = hm.get_height_map();
    let sample_extent =
        Extent2i::from_min_and_shape(Point2i::fill(-20), Point2i::fill(40)).padded(1);

    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3x1 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let builder = ChunkMapBuilder2x1::new(ChunkMapConfig { chunk_shape: PointN([16; 2]), ambient_value: 0.0, root_lod: 0 });
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(&sample_extent, &Func(height_map), &mut map.lod_view_mut(0));

    // Generate the chunk meshes.
    let map_ref = &map;

    pool.scope(|s| {
        for chunk_key in map_ref.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_height_map_chunk_extent(
                    &map_ref.indexer.extent_for_chunk_with_min(chunk_key.minimum),
                )
                // Ignore the ambient values outside the sample extent.
                .intersection(&sample_extent);

                let mut padded_chunk = Array2x1::fill(padded_chunk_extent, 0.0);
                copy_extent(
                    &padded_chunk_extent,
                    &map_ref.lod_view(0),
                    &mut padded_chunk,
                );

                let mut height_map_mesh_buffer = HeightMapMeshBuffer::default();
                triangulate_height_map(
                    &padded_chunk,
                    &padded_chunk_extent,
                    &mut height_map_mesh_buffer,
                );
                let mesh = if flat_shaded {
                    height_map_mesh_buffer.mesh.process_for_flat_shading()
                } else {
                    height_map_mesh_buffer.mesh
                };

                if mesh.positions.is_empty() {
                    None
                } else {
                    Some(mesh)
                }
            })
        }
    })
}

fn generate_chunk_meshes_from_blocky(blocky: Blocky, pool: &TaskPool) -> Vec<Option<PosNormMesh>> {
    let voxels = blocky.get_voxels();

    // Chunk up the voxels just to show that meshing across chunks is consistent.
    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3x1 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let builder = ChunkMapBuilder3x1::new(ChunkMapConfig {
        chunk_shape: Point3i::fill(16),
        ambient_value: CubeVoxel::default(),
        root_lod: 0,
    });
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(voxels.extent(), &voxels, &mut map.lod_view_mut(0));

    // Generate the chunk meshes.
    let map_ref = &map;

    pool.scope(|s| {
        for chunk_key in map_ref.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_greedy_quads_chunk_extent(
                    &map_ref.indexer.extent_for_chunk_with_min(chunk_key.minimum),
                );

                let mut padded_chunk = Array3x1::fill(padded_chunk_extent, CubeVoxel(false));
                copy_extent(
                    &padded_chunk_extent,
                    &map_ref.lod_view(0),
                    &mut padded_chunk,
                );

                let mut buffer = GreedyQuadsBuffer::new(
                    padded_chunk_extent,
                    RIGHT_HANDED_Y_UP_CONFIG.quad_groups(),
                );
                greedy_quads(&padded_chunk, &padded_chunk_extent, &mut buffer);

                let mut mesh = PosNormMesh::default();
                for group in buffer.quad_groups.iter() {
                    for quad in group.quads.iter() {
                        group.face.add_quad_to_pos_norm_mesh(&quad, 1.0, &mut mesh);
                    }
                }

                if mesh.is_empty() {
                    None
                } else {
                    Some(mesh)
                }
            })
        }
    })
}
