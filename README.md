# Building Blocks

[![Crates.io](https://img.shields.io/crates/v/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Docs.rs](https://docs.rs/building-blocks/badge.svg)](https://docs.rs/building-blocks)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bonsairobo/building-blocks/blob/master/LICENSE)
[![Crates.io](https://img.shields.io/crates/d/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Discord](https://img.shields.io/discord/770726405557321778.svg?logo=discord&colorB=7289DA)](https://discord.gg/CnTNjwb)

Building Blocks is a voxel library for real-time applications.

![Meshing](https://i.imgur.com/IZwfRHc.gif)

Supported use cases include:

- memory-efficient storage of voxel maps
- voxel map serialization
- generating meshes
  - isosurface
  - cubic / blocky
  - height maps
- accelerated spatial queries
  - ray casting
  - range queries
- procedural generation
  - sampling signed distance fields
  - generating height maps from fractal noise (TODO)
- pathfinding

## Short Code Example

The code below samples a [signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function) and generates a mesh from it.

```rust
use building_blocks::{
    prelude::*,
    mesh::{SurfaceNetsBuffer, surface_nets},
    procgen::signed_distance_fields::sphere,
};

let center = PointN([25.0; 3]);
let radius = 10.0;
let sphere_sdf = sphere(center, radius);

let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([50; 3]));
let mut samples = Array3::fill_with(extent, &sphere_sdf);

let mut mesh_buffer = SurfaceNetsBuffer::new();
surface_nets(&samples, samples.extent(), &mut mesh_buffer);
```

## Learning

The current best way to learn about the library is to read the documentation and
examples.

For the latest stable docs, look [here](https://docs.rs/building_blocks/latest/building_blocks).

For the latest unstable docs, clone the repo and run

```sh
cargo doc --open --all-features
```

There is plentiful documentation with examples.

Take a look in the `examples/` directory to see how Building Blocks can be used
in real applications.

To run the benchmarks (using the "criterion" crate), go to the root of a crate
and run `cargo bench`.

To learn more about the motivations behind the library's design, read about our
[design philosophy and architecture](https://github.com/bonsairobo/building-blocks/blob/master/DESIGN.md)

## Development

We prioritize work according to the [project board](https://github.com/bonsairobo/building-blocks/projects/1).

If you'd like to make a contribution, please first read the
**[contribution guidelines](https://github.com/bonsairobo/building-blocks/blob/master/CONTRIBUTING.md)**.
