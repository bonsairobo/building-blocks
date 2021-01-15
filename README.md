# Building Blocks

[![Crates.io](https://img.shields.io/crates/v/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Docs.rs](https://docs.rs/building-blocks/badge.svg)](https://docs.rs/building-blocks)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bonsairobo/building-blocks/blob/main/LICENSE)
[![Crates.io](https://img.shields.io/crates/d/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Discord](https://img.shields.io/discord/770726405557321778.svg?logo=discord&colorB=7289DA)](https://discord.gg/CnTNjwb)

Building Blocks is a voxel library for real-time applications.

![Meshing](https://i.imgur.com/IZwfRHc.gif)

Features include:

- memory-efficient storage of voxel maps
  - a `ChunkMap` with generic chunk storage
  - LRU-cached storage of compressed chunks
  - compressed serialization format
  - `OctreeSet` bitset structure
- mesh generation
  - isosurface
  - cubic / blocky
  - height map
- accelerated spatial queries
  - ray casting and sphere casting
  - range queries
- procedural generation
  - sampling signed distance fields
  - constructive solid geometry (TODO)
- pathfinding on voxel maps

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

let mut mesh_buffer = SurfaceNetsBuffer::default();
surface_nets(&samples, samples.extent(), &mut mesh_buffer);
```

## Configuration

### Features and WASM

Building Blocks is organized into several crates, some of which are hidden behind features, and some have features themselves,
which get re-exported by the top-level crate.

For example, chunk compression supports two backends out of the box: `Lz4` and `Snappy`.
They are enabled with the "lz4" and "snappy" features. "lz4" is the default, but it relies on a C++ library, so
it's not compatible with WASM. But Snappy is pure Rust, so it can! Just use `default-features = false` and add "snappy"
to you `features` list, like so:

```toml
[dependencies.building-blocks]
version = "0.4.1"
default-features = false
features = ["snappy"]
```

### LTO

It is highly recommended that you enable link-time optimization when using building-blocks. It will improve the performance
of critical algorithms like meshing by up to 2x. Just add this to your Cargo.toml:

```toml
[profile.release]
lto = true
```

## Learning

The current best way to learn about the library is to read the documentation and
examples.

For the latest stable docs, look [here](https://docs.rs/building_blocks/latest/building_blocks).

For the latest unstable docs, clone the repo and run

```sh
cargo doc --open
```

There is plentiful documentation with examples.

Take a look in the `examples/` directory to see how Building Blocks can be used
in real applications.

To run the benchmarks (using the "criterion" crate), go to the root of a crate
and run `cargo bench`.

To learn more about the motivations behind the library's design, read about our
[design philosophy and architecture](https://github.com/bonsairobo/building-blocks/blob/main/DESIGN.md).

## Development

We prioritize work according to the [project board](https://github.com/bonsairobo/building-blocks/projects/1).

If you'd like to make a contribution, please first read the
**[contribution guidelines](https://github.com/bonsairobo/building-blocks/blob/main/CONTRIBUTING.md)**.
