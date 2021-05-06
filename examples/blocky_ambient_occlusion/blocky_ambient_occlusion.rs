use std::collections::HashMap;

use bevy::{
    input::system::exit_on_esc_system,
    pbr::AmbientLight,
    prelude::{shape, *},
    render::{
        mesh::Indices,
        pipeline::{PipelineDescriptor, PrimitiveTopology, RenderPipeline},
        shader::{ShaderStage, ShaderStages},
        wireframe::{WireframeConfig, WireframePlugin},
    },
    wgpu::{WgpuFeature, WgpuFeatures, WgpuOptions},
};
use bevy_fly_camera::{FlyCamera, FlyCameraPlugin};
use building_blocks::core::prelude::*;
use building_blocks::mesh::{
    greedy_quads_with_merge_strategy, GreedyQuadsBuffer, IsOpaque, MergeVoxel, PosNormTexMesh,
    VoxelAOMerger, RIGHT_HANDED_Y_UP_CONFIG,
};
use building_blocks::storage::{
    access_traits::{Get, GetMut},
    Array3x1, IsEmpty,
};
use building_blocks_mesh::{
    get_ao_at_vert, oriented_cube_face_to_cube_face, CubeFace, FaceStrides, FaceVertex,
    OrientedCubeFace, QuadGroup, UnorientedQuad,
};
use building_blocks_storage::{IndexedArray, Local, Stride};

fn main() {
    App::build()
        .insert_resource(WgpuOptions {
            features: WgpuFeatures {
                // The Wireframe requires NonFillPolygonMode feature
                features: vec![WgpuFeature::NonFillPolygonMode],
            },
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .insert_resource(WireframeConfig {
            global: true,
            ..Default::default()
        })
        .add_plugin(FlyCameraPlugin)
        .add_system(exit_on_esc_system.system())
        .add_startup_system(setup.system())
        .add_system(toggle_fly_camera.system())
        .run();
}

/// Basic voxel type with one byte of texture layers
#[derive(Default, Clone, Copy)]
struct Voxel(bool);

impl MergeVoxel for Voxel {
    type VoxelValue = bool;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}

fn cube_face_to_quad_corners_as_face_vertices(cube_face: CubeFace) -> [FaceVertex; 4] {
    match cube_face {
        CubeFace::Top => [
            FaceVertex::TopLeft,
            FaceVertex::BottomLeft,
            FaceVertex::TopRight,
            FaceVertex::BottomRight,
        ],
        CubeFace::Bottom => [
            FaceVertex::TopRight,
            FaceVertex::BottomRight,
            FaceVertex::TopLeft,
            FaceVertex::BottomLeft,
        ],
        CubeFace::Left => [
            FaceVertex::BottomLeft,
            FaceVertex::BottomRight,
            FaceVertex::TopLeft,
            FaceVertex::TopRight,
        ],
        CubeFace::Right => [
            FaceVertex::BottomRight,
            FaceVertex::BottomLeft,
            FaceVertex::TopRight,
            FaceVertex::TopLeft,
        ],
        CubeFace::Back => [
            FaceVertex::BottomRight,
            FaceVertex::BottomLeft,
            FaceVertex::TopRight,
            FaceVertex::TopLeft,
        ],
        CubeFace::Front => [
            FaceVertex::BottomLeft,
            FaceVertex::BottomRight,
            FaceVertex::TopLeft,
            FaceVertex::TopRight,
        ],
    }
}

#[derive(Debug, Clone)]
struct MeshBuf {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub ambient_occlusion: Vec<f32>,
    pub indices: Vec<u32>,
    pub extent: Extent3i,
}

impl Default for MeshBuf {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            tex_coords: Vec::new(),
            ambient_occlusion: Vec::new(),
            indices: Vec::new(),
            extent: Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([0, 0, 0])),
        }
    }
}

impl MeshBuf {
    fn add_quad(
        &mut self,
        face: &OrientedCubeFace,
        quad: &UnorientedQuad,
        voxel_size: f32,
        u_flip_face: Axis3,
        ambient_occlusions: &[f32; 4],
    ) {
        let start_index = self.positions.len() as u32;
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(quad, voxel_size));
        self.normals.extend_from_slice(&face.quad_mesh_normals());

        let flip_v = true;
        self.tex_coords
            .extend_from_slice(&face.tex_coords(u_flip_face, flip_v, quad));

        self.ambient_occlusion.extend_from_slice(ambient_occlusions);
        self.indices.extend_from_slice(&face.quad_mesh_indices(
            start_index,
            false, // FIXME: Fix flipping! It will need to be different logic for different faces!!!
                   //ambient_occlusions[0] + ambient_occlusions[3]
                   //> ambient_occlusions[1] + ambient_occlusions[2],
        ));
    }
}

fn cube_face_to_face_strides<A>(voxels: &A, cube_face: CubeFace) -> FaceStrides
where
    A: IndexedArray<[i32; 3]>,
{
    let (xs, ys, zs) = (
        voxels.stride_from_local_point(Local(PointN([1, 0, 0]))),
        voxels.stride_from_local_point(Local(PointN([0, 1, 0]))),
        voxels.stride_from_local_point(Local(PointN([0, 0, 1]))),
    );
    match cube_face {
        CubeFace::Top => FaceStrides {
            n_stride: ys,
            u_stride: zs,
            v_stride: xs,
            visibility_offset: ys,
        },
        CubeFace::Bottom => FaceStrides {
            n_stride: ys,
            u_stride: zs,
            v_stride: xs,
            visibility_offset: Stride(0) - ys,
        },
        CubeFace::Left => FaceStrides {
            n_stride: xs,
            u_stride: zs,
            v_stride: ys,
            visibility_offset: Stride(0) - xs,
        },
        CubeFace::Right => FaceStrides {
            n_stride: xs,
            u_stride: zs,
            v_stride: ys,
            visibility_offset: xs,
        },
        CubeFace::Back => FaceStrides {
            n_stride: zs,
            u_stride: xs,
            v_stride: ys,
            visibility_offset: Stride(0) - zs,
        },
        CubeFace::Front => FaceStrides {
            n_stride: zs,
            u_stride: xs,
            v_stride: ys,
            visibility_offset: zs,
        },
    }
}

fn setup(
    mut commands: Commands,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let voxels = set_up_voxels();

    let mut greedy_buffer =
        GreedyQuadsBuffer::new(*voxels.extent(), RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
    greedy_quads_with_merge_strategy::<_, _, VoxelAOMerger<Voxel>>(
        &voxels,
        voxels.extent(),
        &mut greedy_buffer,
    );

    let flip_v = true;
    let voxel_size = 1.0;
    let mut mesh_buf = MeshBuf::default();
    for group in greedy_buffer.quad_groups.iter() {
        let cube_face = oriented_cube_face_to_cube_face(group.face);
        let face_strides = cube_face_to_face_strides(&voxels, cube_face);
        for quad in group.quads.iter() {
            let stride =
                voxels.stride_from_local_point(Local(quad.minimum - voxels.extent().minimum));
            println!("quad corners: {:?}", group.face.quad_corners(quad));
            let corners = cube_face_to_quad_corners_as_face_vertices(
                oriented_cube_face_to_cube_face(group.face),
            );
            // FIXME - THESE HAVE TO USE THE QUAD WIDTH AND QUAD HEIGHT!!!
            let ao_values = [
                get_ao_at_vert(
                    &voxels,
                    &face_strides,
                    stride,
                    cube_face,
                    corners[0], // FIXME: SHOULD BE UV 0,0
                ) as f32,
                get_ao_at_vert(
                    &voxels,
                    &face_strides,
                    stride,
                    cube_face,
                    corners[1], // FIXME: SHOULD BE UV 1,0
                ) as f32,
                get_ao_at_vert(
                    &voxels,
                    &face_strides,
                    stride,
                    cube_face,
                    corners[2], // FIXME: SHOULD BE UV 0,1
                ) as f32,
                get_ao_at_vert(
                    &voxels,
                    &face_strides,
                    stride,
                    cube_face,
                    corners[3], // FIXME: SHOULD BE UV 1,1
                ) as f32,
            ];
            mesh_buf.add_quad(
                &group.face,
                quad,
                voxel_size,
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                &ao_values,
            );
        }
    }

    // Create the bevy mesh
    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let MeshBuf {
        positions,
        normals,
        tex_coords,
        indices,
        ambient_occlusion,
        ..
    } = mesh_buf;

    render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.set_attribute("Vertex_AO", ambient_occlusion);
    render_mesh.set_indices(Some(Indices::U32(indices)));

    // Configure the custom PBR shader with ambient occlusion technique
    let pipeline = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: shaders.add(Shader::from_glsl(
            ShaderStage::Vertex,
            include_str!("ambient_occlusion.vert"),
        )),
        fragment: Some(shaders.add(Shader::from_glsl(
            ShaderStage::Fragment,
            include_str!("ambient_occlusion.frag"),
        ))),
    }));
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(StandardMaterial {
            base_color: Color::SEA_GREEN,
            ..Default::default()
        }),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(pipeline)]),
        ..Default::default()
    });

    // Axes
    let axis_mesh = meshes.add(shape::Cube { size: 1.0 }.into());
    let axis_length = 50.0;
    commands.spawn_bundle(PbrBundle {
        mesh: axis_mesh.clone(),
        material: materials.add(StandardMaterial {
            base_color: Color::RED,
            ..Default::default()
        }),
        transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
            Vec3::new(axis_length, 0.1, 0.1),
            Quat::IDENTITY,
            Vec3::new(0.5 * axis_length, 0.0, 0.0),
        )),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: axis_mesh.clone(),
        material: materials.add(StandardMaterial {
            base_color: Color::GREEN,
            ..Default::default()
        }),
        transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
            Vec3::new(0.1, axis_length, 0.1),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.5 * axis_length, 0.0),
        )),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: axis_mesh,
        material: materials.add(StandardMaterial {
            base_color: Color::BLUE,
            ..Default::default()
        }),
        transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
            Vec3::new(0.1, 0.1, axis_length),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.5 * axis_length),
        )),
        ..Default::default()
    });

    // NOTE: Ambient occlusion only applies to diffuse and specular scattering of ambient light
    // other light sources will easily overpower it so this demo only uses ambient light for the
    // sake of focusing on the ambient occlusion
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.6,
    });
    let xy_offset = Vec3::new(0.1 * axis_length, 0.1 * axis_length, 0.0);
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_matrix(Mat4::face_toward(
                Vec3::new(0.0, 0.0, 0.5 * axis_length) + xy_offset,
                xy_offset,
                Vec3::Y,
            )),
            ..Default::default()
        })
        .insert(FlyCamera::default());
}

const CUBE_FACES: [CubeFace; 6] = [
    CubeFace::Top,
    CubeFace::Front,
    CubeFace::Right,
    CubeFace::Bottom,
    CubeFace::Back,
    CubeFace::Left,
];

const FACE_VERTICES: [FaceVertex; 4] = [
    FaceVertex::TopLeft,
    FaceVertex::TopRight,
    FaceVertex::BottomRight,
    FaceVertex::BottomLeft,
];

fn populate_and_verify(
    voxels: &mut Array3x1<Voxel>,
    positions: &[Point3i],
    offset: Point3i,
    vertex_aos: &HashMap<(Point3i, CubeFace, FaceVertex), i32>,
) {
    for position in positions {
        *voxels.get_mut(offset + *position) = Voxel(true);
    }
    for position in positions {
        let stride =
            voxels.stride_from_local_point(Local(offset + *position - voxels.extent().minimum));
        for cube_face in &CUBE_FACES {
            let face_strides = cube_face_to_face_strides(&*voxels, *cube_face);
            for face_vertex in &FACE_VERTICES {
                assert!(
                    get_ao_at_vert(&*voxels, &face_strides, stride, *cube_face, *face_vertex)
                        == *vertex_aos
                            .get(&(*position, *cube_face, *face_vertex))
                            .unwrap_or(&3)
                );
            }
        }
    }
}

fn set_up_voxels() -> Array3x1<Voxel> {
    println!(">>> set_up_voxels");
    let interior_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([64; 3]));
    let full_extent = interior_extent.padded(1);
    let mut voxels = Array3x1::fill(full_extent, Voxel::default());

    let row_step = PointN([0, 5, 0]);
    let col_step = PointN([5, 0, 0]);
    let mut offset = PointN([1, 1, 1]);

    // .
    let positions = [PointN([1, 1, 1])];
    populate_and_verify(&mut voxels, &positions, offset, &HashMap::new());
    offset += row_step;

    // .  .  .
    //  . . .
    let positions = [PointN([1, 1, 1]), PointN([0, 2, 1])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 1, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += col_step;

    let positions = [PointN([1, 1, 1]), PointN([0, 2, 0])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        (
            (PointN([0, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += col_step;

    let positions = [PointN([1, 1, 1]), PointN([1, 2, 0])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    *offset.x_mut() = 1;
    offset += row_step;

    //  .. ..
    //  .   .
    let positions = [PointN([1, 1, 1]), PointN([0, 2, 0]), PointN([0, 2, 1])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        (
            (PointN([1, 1, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 1),
        (
            (PointN([0, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ), // NOTE: This is a hidden face
        (
            (PointN([0, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += col_step;

    let positions = [PointN([1, 1, 1]), PointN([0, 2, 0]), PointN([1, 2, 0])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        (
            (PointN([0, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This is a hidden face
        (
            (PointN([0, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    *offset.x_mut() = 1;
    offset += row_step;

    //  . ..
    // .. ..
    let positions = [PointN([1, 1, 1]), PointN([0, 2, 1]), PointN([1, 2, 0])];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 0),
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        (
            (PointN([1, 1, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([0, 2, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        (
            (PointN([0, 2, 1]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 2, 0]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            0,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 2, 0]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([1, 2, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += col_step;

    let positions = [
        PointN([1, 1, 1]),
        PointN([0, 2, 0]),
        PointN([0, 2, 1]),
        PointN([1, 2, 0]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 0),
        ((PointN([1, 1, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        (
            (PointN([1, 1, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 1),
        ((PointN([1, 1, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([0, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([0, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([0, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 2, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([0, 2, 1]), CubeFace::Back, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([0, 2, 1]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 2, 0]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            0,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 2, 0]), CubeFace::Left, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    *offset.x_mut() = 1;
    offset += row_step;

    // Looks like a 2-seater sofa and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    let positions = [
        // Left armrest
        PointN([0, 1, 1]),
        // Seat
        PointN([1, 0, 1]),
        PointN([2, 0, 1]),
        // Back
        PointN([1, 1, 0]),
        PointN([2, 1, 0]),
        // Right armrest
        PointN([3, 1, 1]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        // Left armrest
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([0, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        (
            (PointN([0, 1, 1]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        // Left seat
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 0),
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([1, 0, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        // Right seat
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 0),
        (
            (PointN([2, 0, 1]), CubeFace::Top, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([2, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Left back
        ((PointN([1, 1, 0]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            0,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([1, 1, 0]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        // Right back
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([2, 1, 0]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Right arm rest
        (
            (PointN([3, 1, 1]), CubeFace::Bottom, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([3, 1, 1]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([3, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([3, 1, 1]), CubeFace::Back, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([3, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        (
            (PointN([3, 1, 1]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([3, 1, 1]), CubeFace::Left, FaceVertex::BottomLeft),
            0,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += PointN([6, 0, 0]);

    // Looks like a 4-seater sofa and we want to check that the faces of the seat and backrests are
    // merged for the center two seats but not the left/right seats as the ambient occlusion values
    // on the edges are different than in the middle
    let positions = [
        // Left armrest
        PointN([0, 1, 1]),
        // Seat
        PointN([1, 0, 1]),
        PointN([2, 0, 1]),
        PointN([3, 0, 1]),
        PointN([4, 0, 1]),
        // Back
        PointN([1, 1, 0]),
        PointN([2, 1, 0]),
        PointN([3, 1, 0]),
        PointN([4, 1, 0]),
        // Right armrest
        PointN([5, 1, 1]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        // Left armrest
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([0, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        (
            (PointN([0, 1, 1]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        // Left seat
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 0),
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([1, 0, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        // Second seat
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopRight), // NOTE: This face is hidden.
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([2, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Third seat
        ((PointN([3, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([3, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([3, 0, 1]), CubeFace::Right, FaceVertex::TopRight), // NOTE: This face is hidden.
            2,
        ),
        ((PointN([3, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([3, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([3, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Right seat
        ((PointN([4, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([4, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 0),
        (
            (PointN([4, 0, 1]), CubeFace::Top, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([4, 0, 1]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([4, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        ((PointN([4, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([4, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([4, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Left back
        ((PointN([1, 1, 0]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            0,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([1, 1, 0]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        // Second back
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Third back
        (
            (PointN([3, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([3, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Right back
        (
            (PointN([4, 1, 0]), CubeFace::Front, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([4, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([4, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([4, 1, 0]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([4, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([4, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([4, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([4, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Right arm rest
        (
            (PointN([5, 1, 1]), CubeFace::Bottom, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([5, 1, 1]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([5, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([5, 1, 1]), CubeFace::Back, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([5, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        (
            (PointN([5, 1, 1]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([5, 1, 1]), CubeFace::Left, FaceVertex::BottomLeft),
            0,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += PointN([8, 0, 0]);

    // Looks like a 2-seater sofa without arm rests and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    let positions = [
        // Seat
        PointN([0, 0, 1]),
        PointN([1, 0, 1]),
        // Back
        PointN([0, 1, 0]),
        PointN([1, 1, 0]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        // Left seat
        ((PointN([0, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        ((PointN([0, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([0, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([0, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([0, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        // Right seat
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Left back
        (
            (PointN([0, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([0, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        // Right back
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += PointN([4, 0, 0]);

    // Looks like a 4-seater sofa without arm rests and we want to check that the faces of the seat and backrests are
    // merged for the center two seats but not the left/right seats as the ambient occlusion values
    // on the edges are different than in the middle
    let positions = [
        // Seat
        PointN([0, 0, 1]),
        PointN([1, 0, 1]),
        PointN([2, 0, 1]),
        PointN([3, 0, 1]),
        // Back
        PointN([0, 1, 0]),
        PointN([1, 1, 0]),
        PointN([2, 1, 0]),
        PointN([3, 1, 0]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        // Left seat
        ((PointN([0, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        ((PointN([0, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([0, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([0, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([0, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        // Second seat
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([1, 0, 1]), CubeFace::Right, FaceVertex::TopRight), // NOTE: This face is hidden.
            2,
        ),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Third seat
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopRight), // NOTE: This face is hidden.
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([2, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Right seat
        ((PointN([3, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([3, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        ((PointN([3, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([3, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([3, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        // Left back
        (
            (PointN([0, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([0, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        // Second back
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Third back
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Right back
        (
            (PointN([3, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([3, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    *offset.x_mut() = 1;
    offset += row_step;

    // Looks like a 2-seater sofa with high back and large seat and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    let positions = [
        // Left armrest
        PointN([0, 1, 1]),
        // Seat
        PointN([1, 0, 1]),
        PointN([2, 0, 1]),
        PointN([1, 0, 2]),
        PointN([2, 0, 2]),
        PointN([1, 0, 3]),
        PointN([2, 0, 3]),
        // Back
        PointN([1, 1, 0]),
        PointN([2, 1, 0]),
        PointN([1, 2, 0]),
        PointN([2, 2, 0]),
        PointN([1, 3, 0]),
        PointN([2, 3, 0]),
        // Right armrest
        PointN([3, 1, 1]),
    ];
    let vertex_aos: HashMap<(Point3i, CubeFace, FaceVertex), i32> = [
        // Left armrest
        ((PointN([0, 1, 1]), CubeFace::Top, FaceVertex::TopRight), 2),
        (
            (PointN([0, 1, 1]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::TopRight),
            1,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            1,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([0, 1, 1]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([0, 1, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        (
            (PointN([0, 1, 1]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        // Left seat
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 0),
        ((PointN([1, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 1),
        (
            (PointN([1, 0, 1]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 0, 1]), CubeFace::Front, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 1]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 1]), CubeFace::Right, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 1]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 1),
        ((PointN([1, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([1, 0, 1]), CubeFace::Left, FaceVertex::TopRight), 2),
        // Right seat
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopLeft), 1),
        ((PointN([2, 0, 1]), CubeFace::Top, FaceVertex::TopRight), 0),
        (
            (PointN([2, 0, 1]), CubeFace::Top, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 0, 1]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 0, 1]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([2, 0, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        ((PointN([2, 0, 1]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        ((PointN([2, 0, 1]), CubeFace::Left, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([2, 0, 1]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Left seat first extension forward
        ((PointN([1, 0, 2]), CubeFace::Top, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 0, 2]), CubeFace::Front, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 2]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 2]), CubeFace::Right, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 2]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 2]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 2]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 2]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([1, 0, 2]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([1, 0, 2]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 0, 2]), CubeFace::Left, FaceVertex::TopLeft), 2),
        // Right seat first extension forward
        ((PointN([2, 0, 2]), CubeFace::Top, FaceVertex::TopRight), 2),
        ((PointN([2, 0, 2]), CubeFace::Front, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([2, 0, 2]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 0, 2]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([2, 0, 2]), CubeFace::Back, FaceVertex::TopLeft), 2),
        ((PointN([2, 0, 2]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([2, 0, 2]), CubeFace::Back, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([2, 0, 2]), CubeFace::Left, FaceVertex::TopLeft), 2),
        ((PointN([2, 0, 2]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([2, 0, 2]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([2, 0, 2]), CubeFace::Left, FaceVertex::BottomLeft),
            2,
        ),
        // Left seat second extension forward
        (
            (PointN([1, 0, 3]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 0, 3]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 0, 3]), CubeFace::Back, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 0, 3]), CubeFace::Back, FaceVertex::BottomLeft),
            2,
        ),
        // Right seat second extension forward
        ((PointN([2, 0, 3]), CubeFace::Back, FaceVertex::TopRight), 2),
        (
            (PointN([2, 0, 3]), CubeFace::Back, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([2, 0, 3]), CubeFace::Left, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 0, 3]), CubeFace::Left, FaceVertex::BottomLeft),
            2,
        ),
        // Left back
        ((PointN([1, 1, 0]), CubeFace::Top, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Top, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        ((PointN([1, 1, 0]), CubeFace::Front, FaceVertex::TopLeft), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            0,
        ),
        ((PointN([1, 1, 0]), CubeFace::Right, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([1, 1, 0]), CubeFace::Left, FaceVertex::TopRight), 2),
        (
            (PointN([1, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        // Right back
        ((PointN([2, 1, 0]), CubeFace::Top, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 1, 0]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomRight),
            0,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            1,
        ),
        ((PointN([2, 1, 0]), CubeFace::Right, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 1, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([2, 1, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([2, 1, 0]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        ((PointN([2, 1, 0]), CubeFace::Left, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([2, 1, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        // Left back first extension upward
        ((PointN([1, 2, 0]), CubeFace::Top, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Top, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([1, 2, 0]), CubeFace::Right, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Right, FaceVertex::TopRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([1, 2, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ),
        // Right back first extension upward
        ((PointN([2, 2, 0]), CubeFace::Top, FaceVertex::TopLeft), 2),
        (
            (PointN([2, 2, 0]), CubeFace::Top, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 2, 0]), CubeFace::Front, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([2, 2, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([2, 2, 0]), CubeFace::Bottom, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([2, 2, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        (
            (PointN([2, 2, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        ((PointN([2, 2, 0]), CubeFace::Left, FaceVertex::TopLeft), 2), // NOTE: This face is hidden.
        ((PointN([2, 2, 0]), CubeFace::Left, FaceVertex::TopRight), 2), // NOTE: This face is hidden.
        (
            (PointN([2, 2, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 2, 0]), CubeFace::Left, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        // Left back second extension upward
        (
            (PointN([1, 3, 0]), CubeFace::Right, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 3, 0]), CubeFace::Right, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([1, 3, 0]), CubeFace::Bottom, FaceVertex::TopLeft),
            2,
        ),
        (
            (PointN([1, 3, 0]), CubeFace::Bottom, FaceVertex::BottomLeft),
            2,
        ),
        // Right back second extension upward
        (
            (PointN([2, 3, 0]), CubeFace::Left, FaceVertex::BottomRight),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 3, 0]), CubeFace::Left, FaceVertex::BottomLeft),
            2,
        ), // NOTE: This face is hidden.
        (
            (PointN([2, 3, 0]), CubeFace::Bottom, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([2, 3, 0]), CubeFace::Bottom, FaceVertex::BottomRight),
            2,
        ),
        // Right arm rest
        ((PointN([3, 1, 1]), CubeFace::Top, FaceVertex::TopLeft), 2),
        (
            (PointN([3, 1, 1]), CubeFace::Front, FaceVertex::BottomLeft),
            2,
        ),
        (
            (PointN([3, 1, 1]), CubeFace::Bottom, FaceVertex::TopRight),
            2,
        ),
        (
            (PointN([3, 1, 1]), CubeFace::Bottom, FaceVertex::BottomRight),
            1,
        ),
        ((PointN([3, 1, 1]), CubeFace::Back, FaceVertex::TopRight), 1),
        (
            (PointN([3, 1, 1]), CubeFace::Back, FaceVertex::BottomRight),
            2,
        ),
        ((PointN([3, 1, 1]), CubeFace::Left, FaceVertex::TopLeft), 1),
        (
            (PointN([3, 1, 1]), CubeFace::Left, FaceVertex::BottomRight),
            1,
        ),
        (
            (PointN([3, 1, 1]), CubeFace::Left, FaceVertex::BottomLeft),
            0,
        ),
    ]
    .iter()
    .cloned()
    .collect();
    populate_and_verify(&mut voxels, &positions, offset, &vertex_aos);
    offset += PointN([6, 0, 0]);

    // Looks like a 4-seater sofa and we want to check that the faces of the seat and backrests are
    // merged for the center two seats but not the left/right seats as the ambient occlusion values
    // on the edges are different than in the middle
    // Left armrest
    *voxels.get_mut(offset + PointN([0, 1, 1])) = Voxel(true);
    // Seat
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 3])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 3])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 3])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 0, 3])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 3, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 3, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 3, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 3, 0])) = Voxel(true);
    // Right armrest
    *voxels.get_mut(offset + PointN([5, 1, 1])) = Voxel(true);
    offset += PointN([8, 0, 0]);

    // Looks like a 2-seater sofa and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    // Seat
    *voxels.get_mut(offset + PointN([0, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 2])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([0, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    offset += PointN([4, 0, 0]);

    // Looks like a 4-seater sofa and we want to check that the faces of the seat and backrests are
    // merged for the center two seats but not the left/right seats as the ambient occlusion values
    // on the edges are different than in the middle
    // Seat
    *voxels.get_mut(offset + PointN([0, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 2])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([0, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 2, 0])) = Voxel(true);
    *offset.x_mut() = 1;
    offset += row_step;

    println!("<<< set_up_voxels");
    voxels
}

fn toggle_fly_camera(keyboard_input: Res<Input<KeyCode>>, mut fly_camera: Query<&mut FlyCamera>) {
    if keyboard_input.just_pressed(KeyCode::C) {
        for mut fc in fly_camera.iter_mut() {
            fc.enabled = !fc.enabled;
        }
    }
}
