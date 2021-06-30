# Examples

The `examples/` directory is not in the `building-blocks` cargo workspace, so to run an example, you need to `cd examples`
before running `cargo run --example foo`.

All examples use the Bevy engine. This list is roughly ordered by complexity.

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

![Mesh Showcase](/examples/screenshots/mesh_showcase.gif)

## Quad Mesh UVs

A `greedy_quads` mesh of a cube with UV coordinates mapped on all faces. This is useful for seeing that textures are oriented
correctly.

```sh
cargo run --example quad_mesh_uvs
```

![Quad Mesh UVs](/examples/screenshots/quad_mesh_uvs.png)

## Array Texture Materials

Shows how to use an "array texture" to give each type of a voxel a different material.

```sh
cargo run --example array_texture_materials
```

![Array Texture Materials](/examples/screenshots/array_texture_materials.png)

## LOD Terrain

A larger scale example of terrain generated with 3D fractional brownian motion
([SIMDnoise](https://docs.rs/simdnoise/3.1.6/simdnoise/) crate). The further chunks are from the camera, the more they get
downsampled. Chunk meshes are dynamically generated as the camera moves.

You can run with either a blocky map or a smooth map.
```sh
cargo run --example lod_terrain blocky
cargo run --example lod_terrain smooth
```

![LOD Terrain](/examples/screenshots/lod_terrain.png)

## Official Related Projects

- [feldspar](https://github.com/bonsairobo/feldspar): A smooth voxel plugin for Bevy Engine
- [feldspar-editor](https://github.com/bonsairobo/feldspar-editor): A map editor for feldspar

## Community Projects Using building-blocks

- [Colonize](https://github.com/indiv0/colonize): A 3D web game similar to Dwarf Fortress
- [Counterproduction](https://github.com/Counterproduction-game/Counterproduction): A game about competitive spaceship building
- [Minkraft](https://github.com/superdump/minkraft): A Minecraft clone
