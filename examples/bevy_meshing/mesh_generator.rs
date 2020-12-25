use building_blocks::core::prelude::*;
use building_blocks::mesh::*;
use building_blocks::procgen::signed_distance_fields::*;
use building_blocks::storage::{prelude::*, IsEmpty};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
    tasks::{ComputeTaskPool, TaskPool},
};

pub struct MeshGeneratorState {
    current_shape_index: i32,
    chunk_mesh_entities: Vec<Entity>,
}

impl MeshGeneratorState {
    pub fn new() -> Self {
        Self {
            current_shape_index: 0,
            chunk_mesh_entities: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Sdf(Sdf),
    HeightMap(HeightMap),
    Cubic(Cubic),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sdf {
    Cube,
    Plane,
    Sphere,
    Torus,
}

impl Sdf {
    fn get_sdf(&self) -> Box<dyn Fn(&Point3i) -> f32> {
        match self {
            Sdf::Cube => Box::new(cube(PointN([0.0, 0.0, 0.0]), 20.0)),
            Sdf::Plane => Box::new(plane(PointN([0.5, 0.5, 0.5]), 1.0)),
            Sdf::Sphere => Box::new(sphere(PointN([0.0, 0.0, 0.0]), 20.0)),
            Sdf::Torus => Box::new(torus(PointN([16.0, 4.0]))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeightMap {
    Wave,
}

impl HeightMap {
    fn get_height_map(&self) -> impl Fn(&Point2i) -> f32 {
        match self {
            HeightMap::Wave => {
                |p: &Point2i| 10.0 * (1.0 + (0.2 * p.x() as f32).cos() + (0.2 * p.y() as f32).sin())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Cubic {
    Terrace,
}

impl Cubic {
    fn get_voxels(&self) -> Array3<CubeVoxel> {
        match self {
            Cubic::Terrace => {
                let extent =
                    Extent3i::from_min_and_shape(PointN([-20; 3]), PointN([40; 3])).padded(1);
                let mut voxels = Array3::fill(extent, CubeVoxel(false));
                for i in 0..40 {
                    let level = Extent3i::from_min_and_shape(
                        PointN([i - 20; 3]),
                        PointN([40 - i, 1, 40 - i]),
                    );
                    voxels.fill_extent(&level, CubeVoxel(true));
                }

                voxels
            }
        }
    }
}

#[derive(Clone, Copy)]
struct CubeVoxel(bool);

impl MaterialVoxel for CubeVoxel {
    type Material = u8;

    fn material(&self) -> Self::Material {
        1 // only 1 material
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
        1 => Shape::Sdf(Sdf::Plane),
        2 => Shape::Sdf(Sdf::Sphere),
        3 => Shape::Sdf(Sdf::Torus),
        4 => Shape::HeightMap(HeightMap::Wave),
        5 => Shape::Cubic(Cubic::Terrace),
        _ => panic!("bad shape index"),
    }
}

#[derive(Default)]
pub struct MeshMaterial(pub Handle<StandardMaterial>);

pub fn mesh_generator_system(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
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

    if new_shape_requested || state.chunk_mesh_entities.is_empty() {
        // Delete the old meshes.
        for entity in state.chunk_mesh_entities.drain(..) {
            commands.despawn(entity);
        }

        // Sample the new shape.
        let chunk_meshes = match choose_shape(state.current_shape_index) {
            Shape::Sdf(sdf) => generate_chunk_meshes_from_sdf(sdf, &pool.0),
            Shape::HeightMap(hm) => generate_chunk_meshes_from_height_map(hm, &pool.0),
            Shape::Cubic(cubic) => generate_chunk_meshes_from_cubic(cubic, &pool.0),
        };

        for mesh in chunk_meshes.into_iter() {
            if let Some(mesh) = mesh {
                state.chunk_mesh_entities.push(create_mesh_entity(
                    mesh,
                    &mut commands,
                    material.0.clone(),
                    &mut meshes,
                ));
            }
        }
    }
}

const CHUNK_SIZE: i32 = 16;

fn generate_chunk_meshes_from_sdf(sdf: Sdf, pool: &TaskPool) -> Vec<Option<PosNormMesh>> {
    let sdf = sdf.get_sdf();
    let sample_extent = Extent3i::from_min_and_shape(PointN([-20; 3]), PointN([40; 3])).padded(1);

    let builder = ChunkMapBuilder {
        chunk_shape: PointN([CHUNK_SIZE; 3]),
        ambient_value: std::f32::MAX, // air
        default_chunk_metadata: (),
    };
    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(&sample_extent, &sdf, &mut map);

    // Generate the chunk meshes.
    let map_ref = &map;

    pool.scope(|s| {
        for chunk_key in map_ref.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_surface_nets_chunk_extent(
                    &map_ref.indexer.extent_for_chunk_at_key(*chunk_key),
                );
                let mut padded_chunk = Array3::fill(padded_chunk_extent, 0.0);
                copy_extent(&padded_chunk_extent, map_ref, &mut padded_chunk);

                // TODO bevy: we could avoid re-allocating the buffers on every call if we had
                // thread-local storage accessible from this task
                let mut surface_nets_buffer = SurfaceNetsBuffer::default();
                surface_nets(
                    &padded_chunk,
                    &padded_chunk_extent,
                    &mut surface_nets_buffer,
                );

                if surface_nets_buffer.mesh.indices.is_empty() {
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
) -> Vec<Option<PosNormMesh>> {
    let height_map = hm.get_height_map();
    let sample_extent = Extent2i::from_min_and_shape(PointN([-20; 2]), PointN([40; 2])).padded(1);

    let builder = ChunkMapBuilder {
        chunk_shape: PointN([CHUNK_SIZE; 2]),
        ambient_value: 0.0, // air
        default_chunk_metadata: (),
    };
    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(&sample_extent, &height_map, &mut map);

    // Generate the chunk meshes.
    let map_ref = &map;

    pool.scope(|s| {
        for chunk_key in map_ref.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_height_map_chunk_extent(
                    &map_ref.indexer.extent_for_chunk_at_key(*chunk_key),
                )
                // Ignore the ambient values outside the sample extent.
                .intersection(&sample_extent);

                let mut padded_chunk = Array2::fill(padded_chunk_extent, 0.0);
                copy_extent(&padded_chunk_extent, map_ref, &mut padded_chunk);

                // TODO bevy: we could avoid re-allocating the buffers on every call if we had
                // thread-local storage accessible from this task
                let mut height_map_mesh_buffer = HeightMapMeshBuffer::default();
                triangulate_height_map(
                    &padded_chunk,
                    &padded_chunk_extent,
                    &mut height_map_mesh_buffer,
                );

                if height_map_mesh_buffer.mesh.indices.is_empty() {
                    None
                } else {
                    Some(height_map_mesh_buffer.mesh)
                }
            })
        }
    })
}

fn generate_chunk_meshes_from_cubic(cubic: Cubic, pool: &TaskPool) -> Vec<Option<PosNormMesh>> {
    let voxels = cubic.get_voxels();

    // Chunk up the voxels just to show that meshing across chunks is consistent.
    let builder = ChunkMapBuilder {
        chunk_shape: PointN([CHUNK_SIZE; 3]),
        ambient_value: CubeVoxel(false),
        default_chunk_metadata: (),
    };
    // Normally we'd keep this map around in a resource, but we don't need to for this specific example. We could also use an
    // Array3 here instead of a ChunkMap3, but we use chunks for educational purposes.
    let mut map = builder.build_with_hash_map_storage();
    copy_extent(voxels.extent(), &voxels, &mut map);

    // Generate the chunk meshes.
    let map_ref = &map;

    pool.scope(|s| {
        for chunk_key in map_ref.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_greedy_quads_chunk_extent(
                    &map_ref.indexer.extent_for_chunk_at_key(*chunk_key),
                );

                let mut padded_chunk = Array3::fill(padded_chunk_extent, CubeVoxel(false));
                copy_extent(&padded_chunk_extent, map_ref, &mut padded_chunk);

                // TODO bevy: we could avoid re-allocating the buffers on every call if we had
                // thread-local storage accessible from this task
                let mut buffer = GreedyQuadsBuffer::new(padded_chunk_extent);
                greedy_quads(&padded_chunk, &padded_chunk_extent, &mut buffer);

                let mut mesh = PosNormMesh::default();
                for group in buffer.quad_groups.iter() {
                    for (quad, _material) in group.quads.iter() {
                        group.face.add_quad_to_pos_norm_mesh(&quad, &mut mesh);
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

fn create_mesh_entity(
    mesh: PosNormMesh,
    commands: &mut Commands,
    material: Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
) -> Entity {
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    let num_vertices = mesh.positions.len();

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(mesh.positions),
    );
    render_mesh.set_attribute("Vertex_Normal", VertexAttributeValues::Float3(mesh.normals));
    render_mesh.set_attribute(
        "Vertex_UV",
        VertexAttributeValues::Float2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(mesh.indices)));

    commands
        .spawn(PbrComponents {
            mesh: meshes.add(render_mesh),
            material,
            ..Default::default()
        })
        .current_entity()
        .unwrap()
}
