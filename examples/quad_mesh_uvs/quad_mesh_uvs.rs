mod camera_rotation;

use bevy_utilities::bevy::{
    asset::LoadState,
    prelude::*,
    render::{
        mesh::Indices,
        pipeline::PrimitiveTopology,
        texture::{AddressMode, SamplerDescriptor},
    },
};
use building_blocks::core::prelude::*;
use building_blocks::mesh::{
    greedy_quads, GreedyQuadsBuffer, IsOpaque, MergeVoxel, PosNormTexMesh, RIGHT_HANDED_Y_UP_CONFIG,
};
use building_blocks::storage::prelude::*;
use camera_rotation::{camera_rotation_system, CameraRotationState};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
        .add_state(AppState::Loading)
        .add_system_set(SystemSet::on_enter(AppState::Loading).with_system(load_assets.system()))
        .add_system_set(SystemSet::on_update(AppState::Loading).with_system(check_loaded.system()))
        .add_system_set(SystemSet::on_enter(AppState::Run).with_system(setup.system()))
        .add_system_set(
            SystemSet::on_update(AppState::Run).with_system(camera_rotation_system.system()),
        )
        .run();
}

fn load_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load("uv_checker.png");
    commands.insert_resource(Loading(handle));
}

/// Make sure that our texture is loaded so we can change some settings on it later
fn check_loaded(
    mut state: ResMut<State<AppState>>,
    handle: Res<Loading>,
    asset_server: Res<AssetServer>,
) {
    if let LoadState::Loaded = asset_server.get_load_state(&handle.0) {
        state.set(AppState::Run).unwrap();
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
    mut commands: Commands,
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

    // Just a solid cube of voxels. We only fill the interior since we need some empty voxels to form a boundary for the mesh.
    let interior_extent = Extent3i::from_min_and_shape(PointN([-10; 3]), PointN([20; 3]));
    let full_extent = interior_extent.padded(1);
    let mut voxels = Array3x1::fill(full_extent, Voxel::default());
    voxels.fill_extent(&interior_extent, Voxel(true));

    let mut greedy_buffer =
        GreedyQuadsBuffer::new(full_extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
    greedy_quads(&voxels, &full_extent, &mut greedy_buffer);

    let flip_v = true;
    let voxel_size = 1.0;
    let mut mesh_buf = PosNormTexMesh::default();
    for group in greedy_buffer.quad_groups.iter() {
        for quad in group.quads.iter() {
            group.face.add_quad_to_pos_norm_tex_mesh(
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                flip_v,
                quad,
                voxel_size,
                &mut mesh_buf,
            );
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

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(texture_handle.0.clone().into()),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 50.0, 50.0)),
        light: Light {
            range: 200.0,
            intensity: 20000.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let camera = commands
        .spawn_bundle(PerspectiveCameraBundle::default())
        .id();

    commands.insert_resource(CameraRotationState::new(camera));
}
