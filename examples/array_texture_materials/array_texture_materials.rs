mod camera_rotation;

use bevy::{
    prelude::*,
    render::{
        mesh::Indices,
        pipeline::{PipelineDescriptor, PrimitiveTopology, RenderPipeline},
        shader::ShaderStages,
        texture::{AddressMode, SamplerDescriptor},
    },
};
use building_blocks::core::prelude::*;
use building_blocks::mesh::{
    greedy_quads, GreedyQuadsBuffer, IsOpaque, MergeVoxel, OrientedCubeFace, UnorientedQuad,
    RIGHT_HANDED_Y_UP_CONFIG,
};
use building_blocks::storage::{Array3x1, Get, IsEmpty};
use camera_rotation::{camera_rotation_system, CameraRotationState};

#[derive(Clone, Copy, Eq, PartialEq)]
enum AppState {
    Loading,
    Run,
}

const TEXTURE_LAYERS: u32 = 4;
const UV_SCALE: f32 = 0.1;

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
    let handle = asset_server.load("materials.png");
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
struct Voxel(u8);

impl MergeVoxel for Voxel {
    type VoxelValue = u8;

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
        self.0 == 0
    }
}

/// Utility struct for building the mesh
#[derive(Debug, Default, Clone)]
struct MeshBuf {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub layer: Vec<u32>,
    pub indices: Vec<u32>,
}

impl MeshBuf {
    fn add_quad(
        &mut self,
        face: &OrientedCubeFace,
        quad: &UnorientedQuad,
        u_flip_face: Axis3,
        layer: u32,
    ) {
        let start_index = self.positions.len() as u32;
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(quad));
        self.normals.extend_from_slice(&face.quad_mesh_normals());

        let flip_v = true;
        let mut uvs = face.tex_coords(u_flip_face, flip_v, quad);
        for uv in uvs.iter_mut() {
            for c in uv.iter_mut() {
                *c *= UV_SCALE;
            }
        }
        self.tex_coords.extend_from_slice(&uvs);

        self.layer.extend_from_slice(&[layer; 4]);
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(start_index));
    }
}

fn setup(
    mut commands: Commands,
    texture_handle: Res<Loading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
) {
    let mut texture = textures.get_mut(&texture_handle.0).unwrap();

    // Set the texture to tile over the entire quad
    texture.sampler = SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        ..Default::default()
    };

    texture.reinterpret_stacked_2d_as_array(TEXTURE_LAYERS);

    // Generate some voxel terrain
    let extent = Extent3i::from_min_and_shape(PointN([-20; 3]), PointN([40; 3])).padded(1);
    let mut voxels = Array3x1::fill(extent, Voxel::default());
    for i in 0..40 {
        let level = Extent3i::from_min_and_shape(PointN([i - 20; 3]), PointN([40 - i, 1, 40 - i]));
        voxels.fill_extent(&level, Voxel((i % 4) as u8 + 1));
    }

    let mut greedy_buffer = GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
    greedy_quads(&voxels, &extent, &mut greedy_buffer);

    let mut mesh_buf = MeshBuf::default();
    for group in greedy_buffer.quad_groups.iter() {
        for quad in group.quads.iter() {
            let mat = voxels.get(quad.minimum);
            mesh_buf.add_quad(
                &group.face,
                quad,
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                mat.0 as u32 - 1,
            );
        }
    }

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let MeshBuf {
        positions,
        normals,
        tex_coords,
        layer,
        indices,
    } = mesh_buf;

    render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.set_attribute("Vertex_Layer", layer);
    render_mesh.set_indices(Some(Indices::U32(indices)));

    let pipeline = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: shaders.add(Shader::from_glsl(
            bevy::render::shader::ShaderStage::Vertex,
            VERTEX_SHADER,
        )),
        fragment: Some(shaders.add(Shader::from_glsl(
            bevy::render::shader::ShaderStage::Fragment,
            FRAGMENT_SHADER,
        ))),
    }));

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(pipeline)]),
        material: materials.add(texture_handle.0.clone().into()),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 100.0, 100.0)),
        ..Default::default()
    });
    let camera = commands
        .spawn_bundle(PerspectiveCameraBundle::default())
        .id();

    commands.insert_resource(CameraRotationState::new(camera));
}

/// Default bevy vertex shader with added vertex attribute for texture layer
const VERTEX_SHADER: &str = r#"
#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;
layout(location = 3) in int Vertex_Layer; // New thing

layout(location = 0) out vec3 v_Position;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec3 v_Uv;

layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
};

layout(set = 2, binding = 0) uniform Transform {
    mat4 Model;
};

void main() {
    v_Normal = mat3(Model) * Vertex_Normal;
    v_Position = (Model * vec4(Vertex_Position, 1.0)).xyz;

    // Gets used here and passed to the fragment shader.
    v_Uv = vec3(Vertex_Uv, Vertex_Layer);

    gl_Position = ViewProj * vec4(v_Position, 1.0);
}
"#;

/// Modified from bevy's default PBR shader.
/// How it works is not really important, the key part is using an array texture 2D
const FRAGMENT_SHADER: &str = r#"
#version 450

const int MAX_LIGHTS = 10;

struct Light {
    mat4 proj;
    vec4 pos;
    vec4 color;
};

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Uv;

layout(location = 0) out vec4 o_Target;

layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
};

layout(set = 1, binding = 0) uniform Lights {
    vec3 AmbientColor;
    uvec4 NumLights;
    Light SceneLights[MAX_LIGHTS];
};

layout(set = 3, binding = 0) uniform StandardMaterial_albedo {
    vec4 Albedo;
};

# ifdef STANDARDMATERIAL_ALBEDO_TEXTURE
layout(set = 3, binding = 1) uniform texture2DArray StandardMaterial_albedo_texture;
layout(set = 3, binding = 2) uniform sampler StandardMaterial_albedo_texture_sampler;
# endif

void main() {
    vec4 output_color = Albedo;
# ifdef STANDARDMATERIAL_ALBEDO_TEXTURE
    // Important part is here
    output_color *= texture(
        sampler2DArray(StandardMaterial_albedo_texture, StandardMaterial_albedo_texture_sampler), 
        v_Uv
    );

# endif

# ifndef STANDARDMATERIAL_UNLIT
    vec3 normal = normalize(v_Normal);
    // accumulate color
    vec3 color = AmbientColor;
    for (int i=0; i<int(NumLights.x) && i<MAX_LIGHTS; ++i) {
        Light light = SceneLights[i];
        // compute Lambertian diffuse term
        vec3 light_dir = normalize(light.pos.xyz - v_Position);
        float diffuse = max(0.0, dot(normal, light_dir));
        // add light contribution
        color += diffuse * light.color.xyz;
    }

    // average the lights so that we will never get something with > 1.0
    color /= max(float(NumLights.x), 1.0);

    output_color.xyz *= color;
# endif

    // multiply the light by material color
    o_Target = output_color;
}
"#;
