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
use building_blocks::storage::{access_traits::{Get, GetMut}, Array3x1, IsEmpty};

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
    let mut mesh_buf = PosNormTexMesh::default();
    let mut ambient_occlusions = Vec::new();
    for group in greedy_buffer.quad_groups.iter() {
        for quad in group.quads.iter() {
            let mut ao_values = [0f32; 4];
            for (i, vertex) in group.face.quad_corners(quad).iter().enumerate() {
                ao_values[i] = get_ao_at_vert_pos(*vertex, &voxels, voxels.extent()) as f32;
            }
            ambient_occlusions.extend_from_slice(&ao_values);
            group.face.add_quad_to_pos_norm_tex_mesh(
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                flip_v,
                quad,
                voxel_size,
                &mut mesh_buf,
            );
        }
    }

    // Create the bevy mesh
    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let PosNormTexMesh {
        positions,
        normals,
        tex_coords,
        indices,
    } = mesh_buf;

    render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.set_attribute("Vertex_AO", ambient_occlusions);
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

fn set_up_voxels() -> Array3x1<Voxel> {
    let interior_extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([64; 3]));
    let full_extent = interior_extent.padded(1);
    let mut voxels = Array3x1::fill(full_extent, Voxel::default());

    let row_step = PointN([0, 5, 0]);
    let col_step = PointN([5, 0, 0]);
    let mut offset = PointN([1, 1, 1]);

    // .
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    offset += row_step;

    // .  .  .
    //  . . .
    *voxels.get_mut(offset + PointN([0, 2, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    offset += col_step;

    *voxels.get_mut(offset + PointN([2, 2, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    offset += col_step;

    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    *offset.x_mut() = 1;
    offset += row_step;

    //  .. ..
    //  .   .
    *voxels.get_mut(offset + PointN([0, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 2, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    offset += col_step;

    *voxels.get_mut(offset + PointN([0, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    *offset.x_mut() = 1;
    offset += row_step;

    //  . ..
    // .. ..
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 2, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    offset += col_step;

    *voxels.get_mut(offset + PointN([0, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([0, 2, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 1])) = Voxel(true);
    *offset.x_mut() = 1;
    offset += row_step;

    // Looks like a 2-seater sofa and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    // Left armrest
    *voxels.get_mut(offset + PointN([0, 1, 1])) = Voxel(true);
    // Seat
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 1])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    // Right armrest
    *voxels.get_mut(offset + PointN([3, 1, 1])) = Voxel(true);
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
    // Back
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([4, 1, 0])) = Voxel(true);
    // Right armrest
    *voxels.get_mut(offset + PointN([5, 1, 1])) = Voxel(true);
    offset += PointN([8, 0, 0]);

    // Looks like a 2-seater sofa and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    // Seat
    *voxels.get_mut(offset + PointN([0, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([0, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    offset += PointN([4, 0, 0]);

    // Looks like a 4-seater sofa and we want to check that the faces of the seat and backrests are
    // merged for the center two seats but not the left/right seats as the ambient occlusion values
    // on the edges are different than in the middle
    // Seat
    *voxels.get_mut(offset + PointN([0, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 0, 1])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([0, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([3, 1, 0])) = Voxel(true);
    *offset.x_mut() = 1;
    offset += row_step;

    // Looks like a 2-seater sofa and we want to check that the faces of the seat and backrests are NOT merged
    // as the ambient occlusion values on the edges are different than in the middle
    // Left armrest
    *voxels.get_mut(offset + PointN([0, 1, 1])) = Voxel(true);
    // Seat
    *voxels.get_mut(offset + PointN([1, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 1])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 2])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 0, 3])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 0, 3])) = Voxel(true);
    // Back
    *voxels.get_mut(offset + PointN([1, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 1, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 2, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([1, 3, 0])) = Voxel(true);
    *voxels.get_mut(offset + PointN([2, 3, 0])) = Voxel(true);
    // Right armrest
    *voxels.get_mut(offset + PointN([3, 1, 1])) = Voxel(true);
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

    voxels
}

fn get_ao_at_vert_pos<A, T>(v: Point3i, padded_chunk: &A, padded_chunk_extent: &Extent3i) -> i32
where
    A: Get<Point3i, Item = T>,
    T: IsEmpty + IsOpaque + MergeVoxel,
{
    let loc: Point3i = v;

    let top0_loc = PointN([loc.x() - 1, loc.y(), loc.z()]);
    let top1_loc: Point3i = PointN([loc.x(), loc.y(), loc.z() - 1]);
    let top2_loc: Point3i = PointN([loc.x(), loc.y(), loc.z()]);
    let top3_loc: Point3i = PointN([loc.x() - 1, loc.y(), loc.z() - 1]);

    let bot0_loc: Point3i = PointN([loc.x() - 1, loc.y() - 1, loc.z()]);
    let bot1_loc: Point3i = PointN([loc.x(), loc.y() - 1, loc.z() - 1]);
    let bot2_loc: Point3i = PointN([loc.x(), loc.y() - 1, loc.z()]);
    let bot3_loc: Point3i = PointN([loc.x() - 1, loc.y() - 1, loc.z() - 1]);

    let top0 = padded_chunk_extent.contains(top0_loc) && !padded_chunk.get(top0_loc).is_empty();
    let top1 = padded_chunk_extent.contains(top1_loc) && !padded_chunk.get(top1_loc).is_empty();
    let top2 = padded_chunk_extent.contains(top2_loc) && !padded_chunk.get(top2_loc).is_empty();
    let top3 = padded_chunk_extent.contains(top3_loc) && !padded_chunk.get(top3_loc).is_empty();
    let bot0 = padded_chunk_extent.contains(bot0_loc) && !padded_chunk.get(bot0_loc).is_empty();
    let bot1 = padded_chunk_extent.contains(bot1_loc) && !padded_chunk.get(bot1_loc).is_empty();
    let bot2 = padded_chunk_extent.contains(bot2_loc) && !padded_chunk.get(bot2_loc).is_empty();
    let bot3 = padded_chunk_extent.contains(bot3_loc) && !padded_chunk.get(bot3_loc).is_empty();

    let (side1, side2, corner) = if !top0 && bot0 {
        (top2, top3, top1)
    } else if !top1 && bot1 {
        (top2, top3, top0)
    } else if !top2 && bot2 {
        (top0, top1, top3)
    } else if !top3 && bot3 {
        (top0, top1, top2)
    } else {
        return 3;
    };

    if side1 && side2 {
        return 0;
    }

    return 3 - (side1 as i32 + side2 as i32 + corner as i32);
}

fn toggle_fly_camera(keyboard_input: Res<Input<KeyCode>>, mut fly_camera: Query<&mut FlyCamera>) {
    if keyboard_input.just_pressed(KeyCode::C) {
        for mut fc in fly_camera.iter_mut() {
            fc.enabled = !fc.enabled;
        }
    }
}
