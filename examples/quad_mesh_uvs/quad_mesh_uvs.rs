mod camera_rotation;

use bevy::{
    prelude::*,
    render::{
        mesh::Indices,
        pipeline::PrimitiveTopology,
        texture::{AddressMode, SamplerDescriptor},
    },
};
use building_blocks::core::prelude::*;
use building_blocks::mesh::{
    greedy_quads, GreedyQuadsBuffer, IsOpaque, MergeVoxel, PosNormTexMesh,
};
use building_blocks::storage::{Array3x1, IsEmpty};
use camera_rotation::{camera_rotation_system, CameraRotationState};

const APP_STAGE: &str = "app_stage";

#[derive(Clone, Copy)]
enum AppState {
    Loading,
    Run,
}

const UV_SCALE: f32 = 1.0 / 16.0;

struct Loading(Handle<Texture>);

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .insert_resource(State::new(AppState::Loading))
        .add_stage_after(
            CoreStage::Update,
            APP_STAGE,
            StateStage::<AppState>::default(),
        )
        .on_state_enter(APP_STAGE, AppState::Loading, load_assets.system())
        .on_state_update(APP_STAGE, AppState::Loading, check_loaded.system())
        .on_state_enter(APP_STAGE, AppState::Run, setup.system())
        .on_state_update(APP_STAGE, AppState::Run, camera_rotation_system.system())
        .run();
}

fn load_assets(commands: &mut Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load("uv_checker.png");
    commands.insert_resource(Loading(handle));
}

/// Make sure that our texture is loaded so we can change some settings on it later
fn check_loaded(
    mut state: ResMut<State<AppState>>,
    handle: Res<Loading>,
    asset_server: Res<AssetServer>,
) {
    if let bevy::asset::LoadState::Loaded = asset_server.get_load_state(&handle.0) {
        state.set_next(AppState::Run).unwrap();
    }
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
    commands: &mut Commands,
    texture_handle: Res<Loading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
) {
    let mut texture = textures.get_mut(&texture_handle.0).unwrap();

    // Set the texture to tile over the entire quad
    texture.sampler = SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        ..Default::default()
    };

    // Generate some voxel terrain
    let interior_extent = Extent3i::from_min_and_shape(PointN([-10; 3]), PointN([20; 3]));
    let full_extent = interior_extent.padded(1);
    let mut voxels = Array3x1::fill(full_extent, Voxel::default());
    voxels.fill_extent(&interior_extent, Voxel(true));

    let mut greedy_buffer = GreedyQuadsBuffer::new_with_y_up(full_extent);
    greedy_quads(&voxels, &full_extent, &mut greedy_buffer);

    let mut mesh_buf = PosNormTexMesh::default();
    for group in greedy_buffer.quad_groups.iter() {
        for quad in group.quads.iter() {
            group
                .face
                .add_quad_to_pos_norm_tex_mesh(true, quad, &mut mesh_buf);
        }
    }

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let PosNormTexMesh {
        positions,
        normals,
        mut tex_coords,
        indices,
    } = mesh_buf;

    for uv in tex_coords.iter_mut() {
        for c in uv.iter_mut() {
            *c *= UV_SCALE;
        }
    }

    render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.set_indices(Some(Indices::U32(indices)));

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(render_mesh),
            material: materials.add(texture_handle.0.clone().into()),
            ..Default::default()
        })
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 100.0, 100.0)),
            ..Default::default()
        });
    let camera = commands
        .spawn(PerspectiveCameraBundle::default())
        .current_entity()
        .unwrap();

    commands.insert_resource(CameraRotationState::new(camera));
}
