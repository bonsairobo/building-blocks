# Examples

## Bevy Minimal

A simple example of how to generate a voxel mesh for use in the Bevy game engine.
Uses sdfu for modeling a signed distance field.

```sh
cargo run --example bevy_minimal
```

![Minimal](https://i.imgur.com/pnTRdO4.png)

## Bevy Meshing

A showcase of all the meshing algorithms, fetching samples from a `ChunkMap`.

```sh
cargo run --example bevy_meshing
```

Use the left and right arrow keys to select one of the example shapes to be meshed.

![Meshing](https://i.imgur.com/IZwfRHc.gif)

## Official Related Projects

All of these are works in progress and likely to break as things change upstream.

- [building-blocks-editor](https://github.com/bonsairobo/building-blocks-editor): A voxel map editor built with Bevy
- [bevy-building-blocks](https://github.com/bonsairobo/bevy-building-blocks): Helpful Bevy plugins for voxel map IO
- [smooth-voxel-renderer](https://github.com/bonsairobo/smooth-voxel-renderer): A Bevy plugin for rendering textured, smooth voxel meshes

## Community Projects Using building-blocks

- [Colonize](https://github.com/indiv0/colonize): A 3D web game similar to Dwarf Fortress
- [Counterproduction](https://github.com/Counterproduction-game/Counterproduction): A game about competitive spaceship building
- [Minkraft](https://github.com/superdump/minkraft): A Minecraft clone
