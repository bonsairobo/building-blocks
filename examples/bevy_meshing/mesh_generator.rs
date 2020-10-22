use building_blocks::core::prelude::*;
use building_blocks::mesh::surface_nets::*;
use building_blocks::procgen::signed_distance_fields::*;
use building_blocks::storage::prelude::*;

use bevy::{
    prelude::*,
    render::{mesh::VertexAttribute, pipeline::PrimitiveTopology},
};

pub struct MeshGeneratorState {
    current_shape: Shape,
    surface_nets_buffer: SurfaceNetsBuffer, // reused to avoid reallocations
    chunk_mesh_entities: Vec<Entity>,
}

impl MeshGeneratorState {
    pub fn new() -> Self {
        Self {
            current_shape: Shape::Cube,
            surface_nets_buffer: SurfaceNetsBuffer::default(),
            chunk_mesh_entities: Vec::new(),
        }
    }
}

#[allow(dead_code)] // some variants are only constructed via transmute
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Cube = 0,
    Plane = 1,
    Sphere = 2,
    Torus = 3,
    Invalid = 4,
}

impl Shape {
    pub fn next(&self) -> Self {
        unsafe { std::mem::transmute((*self as i8 + 1).rem_euclid(Shape::Invalid as i8)) }
    }

    pub fn previous(&self) -> Self {
        unsafe { std::mem::transmute((*self as i8 - 1).rem_euclid(Shape::Invalid as i8)) }
    }

    pub fn get_sdf(self: Shape) -> Box<dyn Fn(&Point3i) -> f32> {
        match self {
            Shape::Cube => Box::new(cube(PointN([0.0, 0.0, 0.0]), 25.0)),
            Shape::Plane => Box::new(plane(PointN([0.5, 0.5, 0.5]), 1.0)),
            Shape::Sphere => Box::new(sphere(PointN([0.0, 0.0, 0.0]), 25.0)),
            Shape::Torus => Box::new(torus(PointN([25.0, 10.0]))),
            Shape::Invalid => panic!("Invalid shape"),
        }
    }
}

#[derive(Default)]
pub struct MeshMaterial(pub Handle<StandardMaterial>);

pub fn mesh_generator_system(
    mut commands: Commands,
    mut state: ResMut<MeshGeneratorState>,
    mut meshes: ResMut<Assets<Mesh>>,
    keyboard_input: Res<Input<KeyCode>>,
    material: Res<MeshMaterial>,
) {
    let mut new_shape_requested = false;
    if keyboard_input.just_pressed(KeyCode::Left) {
        new_shape_requested = true;
        state.current_shape = state.current_shape.previous();
    } else if keyboard_input.just_pressed(KeyCode::Right) {
        new_shape_requested = true;
        state.current_shape = state.current_shape.next();
    }

    if new_shape_requested || state.chunk_mesh_entities.is_empty() {
        // Delete the old meshes.
        for entity in state.chunk_mesh_entities.drain(..) {
            commands.despawn(entity);
        }

        // Sample the new shape SDF in some extent.
        let sdf = state.current_shape.get_sdf();
        let sample_extent = Extent3i::from_min_and_shape(PointN([-50; 3]), PointN([100; 3]));
        let chunk_shape = PointN([16; 3]);
        let ambient_value = std::f32::MAX; // air
        let default_chunk_meta = ();
        // Normally we'd keep this map around in a resource, but we don't need to for this specific
        // example. We could also use an Array3 here instead of a ChunkMap3, but we use chunks for
        // educational purposes.
        let mut map = ChunkMap3::new(
            chunk_shape,
            ambient_value,
            default_chunk_meta,
            FastLz4 { level: 10 },
        );
        copy_extent(&sample_extent, &sdf, &mut map);

        // Generate the chunk meshes.
        let local_cache = LocalChunkCache::new();
        let map_reader = ChunkMapReader3::new(&map, &local_cache);
        for chunk_key in map.chunk_keys() {
            let padded_chunk_extent = map.extent_for_chunk_at_key(chunk_key).padded(1);
            let mut padded_chunk = Array3::fill(padded_chunk_extent, 0.0);
            copy_extent(&padded_chunk_extent, &map_reader, &mut padded_chunk);
            surface_nets(
                &padded_chunk,
                &padded_chunk_extent,
                &mut state.surface_nets_buffer,
            );

            if state.surface_nets_buffer.mesh.indices.is_empty() {
                continue;
            }

            let mesh = meshes.add(Mesh {
                primitive_topology: PrimitiveTopology::TriangleList,
                attributes: vec![
                    VertexAttribute::position(state.surface_nets_buffer.mesh.positions.clone()),
                    VertexAttribute::normal(state.surface_nets_buffer.mesh.normals.clone()),
                    // UVs don't matter for this monocolor mesh
                    VertexAttribute::uv(vec![
                        [0.0; 2];
                        state.surface_nets_buffer.mesh.normals.len()
                    ]),
                ],
                indices: Some(
                    state
                        .surface_nets_buffer
                        .mesh
                        .indices
                        .iter()
                        .map(|i| *i as u32)
                        .collect(),
                ),
            });
            let mesh_entity = commands
                .spawn(PbrComponents {
                    mesh,
                    material: material.0,
                    ..Default::default()
                })
                .current_entity()
                .unwrap();
            state.chunk_mesh_entities.push(mesh_entity);
        }
    }
}
