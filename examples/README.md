# Examples

All examples use the Bevy engine.

## SDF Mesh

A simple example of how to generate a mesh from signed distance voxels.

```sh
cargo run --example sdf_mesh
```

<img src="/examples/screenshots/sdf_mesh.png" alt="SDF Mesh" width="400">

## Mesh Showcase

A showcase of all the meshing algorithms, fetching samples from a `ChunkMap`.

```sh
cargo run --example mesh_showcase
```

Use the left and right arrow keys to select one of the example shapes to be meshed.

![Mesh Showcase](/examples/screenshots/mesh_showcase.png)

## Array Texture Materials

Shows how to use an "array texture" to give each type of a voxel a different material.

```sh
cargo run --example array_texture_materials
```

![Array Texture Materials](/examples/screenshots/mesh_showcase.png)

## Quad Mesh UVs

A `greedy_quads` mesh of a cube with UV coordinates mapped on all faces. This is useful for seeing that textures are oriented
correctly.

```sh
cargo run --example quad_mesh_uvs
```

![Quad Mesh UVs](/examples/screenshots/quad_mesh_uvs.png)

## Official Related Projects

All of these are works in progress and likely to break as things change upstream.

- [building-blocks-editor](https://github.com/bonsairobo/building-blocks-editor): A voxel map editor built with Bevy
- [bevy-building-blocks](https://github.com/bonsairobo/bevy-building-blocks): Helpful Bevy plugins for voxel map IO
- [smooth-voxel-renderer](https://github.com/bonsairobo/smooth-voxel-renderer): A Bevy plugin for rendering textured, smooth voxel meshes

## Community Projects Using building-blocks

- [Colonize](https://github.com/indiv0/colonize): A 3D web game similar to Dwarf Fortress
- [Counterproduction](https://github.com/Counterproduction-game/Counterproduction): A game about competitive spaceship building
- [Minkraft](https://github.com/superdump/minkraft): A Minecraft clone
