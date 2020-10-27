# Building Blocks

[![Crates.io](https://img.shields.io/crates/v/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Docs.rs](https://docs.rs/building-blocks/badge.svg)](https://docs.rs/building-blocks)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bonsairobo/building-blocks/blob/master/LICENSE)
[![Crates.io](https://img.shields.io/crates/d/building-blocks.svg)](https://crates.io/crates/building-blocks)
[![Discord](https://img.shields.io/discord/770726405557321778)](https://discordapp.com/invite/gVMxUm)

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
  - range queries (TODO)
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

## Design Philosophy and Architecture

### Principles

The architecture of Building Blocks is driven by a few guiding principles:

- **Real-Time Performance**
  - The primary use case for Building Blocks is using voxel technology within a
    video game. This means it needs to be fast. For example, we want to be able
    to generate meshes for millions of voxels per frame (16.6 ms).
  - Critical algorithms must be benchmarked with Criterion so we can guide
    optimization with evidence.
- **Composable APIs**
  - APIs are more powerful when they are generic. You will find many examples
    of generic APIs that require the input types to implement some traits.
  - This is most prevalent in the storage crate, where we desire all of the
    lattice map types to be accessible through the same core set of traits.
    This results in a very powerful function, `copy_extent`, which works with
    all implementations of the `ReadExtent` and `WriteExtent` traits.
- **KISS (Keep It Simple Stupid)**
  - There are *many* complex data structures and algorithms used in voxel
    technology. While they certainly all serve a purpose, it is not feasible for
    contributors to understand and implement all of them at the very inception
    of this project.
  - Any increase in complexity of this library, usually by adding a new data
    structure or algorithm, must solve a specific problem for end users. And we
    should find the simplest solution to that problem such that the increase in
    complexity is minimal.
  - You might find your favorite voxel technology missing from the existing
    feature set or road map. This is probably just because no one has found a
    need for it yet. We are open to contributions, but keep the KISS principle
    in mind when considering what problem you are solving.
- **Minimal Dependencies**
  - Rather than taking on large dependencies like nalgebra, ncollide, ndarray,
    etc, we will first try to implement the simplest version of what is needed
    ourselves.
  - Integrations with 3rd party dependencies are exposed under feature flags to
    avoid bloat.
  - There is always a judgement call when determining if we should take on a
    dependency. The main considerations are build time, difficulty of "rolling
    our own," and of course the full feature set and performance of the
    dependency.

### Architecture

Noting the above principles, here is a quick summary of the design decisions
which brought about the current feature set:

- **Mathematical Data Types**
  - A voxel world can be modeled mathematically as a function over a
    3-dimensional integer lattice, i.e. with domain Z^3. Thus we have a
    `Point3i` type which serves as the main coordinate set.
  - When considering a subset of the voxel world, in coordinates, we very
    commonly desire an axis-aligned bounding box, which may contain any bounded
    subset. It is very simple to iterate over all of the points in such a box,
    find the intersection of two boxes, or check if the box contains a point. We
    call these boxes by the name `Extent3i`.
- **Storage**
  - Any extent of the voxel world function can be simply represented as a
    3-dimensional array. We call it `Array3`. Arrays contain some data at every
    point in some `Extent3i`. The most common operations on arrays are iteration
    over extents and random access. These operations should be very fast, and we
    have benchmarks for them.
  - Obviously we can't store an infinite voxel world, so we partition the
    lattice into chunks, each of which is an `Array3` of the same cubic shape.
    The container for these chunks is called a `ChunkMap3`.
  - With both the `Array3` and `ChunkMap3` serving similar purposes, we've made
    them implement a common set of traits for data access. This includes random
    access, iteration, and copying.
  - When you have large voxel worlds, it's not feasible to store a lot of unique
    data for every voxel. A common strategy is to have each voxel labeled with
    some "type." If you only want to use a single byte for each voxel's type,
    then you can have up to 255 types of voxels. Then each type can have a large
    amount of data associated with it in a "palette" container. But we still
    want to be able to use our common set of access traits to read the voxel
    type data. Thus, we have a type called `TransformMap` which implements those
    traits. `TransformMap` wraps some other lattice map, referred to as the
    "delegate," and any access will first go through the delegate, then be
    transformed by an arbitrary `Fn(T) -> S` chosen by the programmer. This
    transformation closure is where we can access the palette based on the voxel
    type provided by the delegate map.
  - Even with only a couple bytes per voxel, we can still use up lots of memory
    on large voxel maps. The simplest way to save memory without changing the
    underlying array containers was to use compression inside of the `ChunkMap`.
    So arrays now support an LZ4 compression scheme. While LZ4 has a very quick
    decompression rate, the `ChunkMap` still keeps an LRU cache of uncompressed
    chunks for efficiency. Cache eviction and compression is done on-demand, so
    you can choose to ignore this feature if you are not worried about memory
    usage.
- **Meshing**
  - There are many ways of generating meshes from voxel data. You can make each
    occupied voxel into a cube, which gives the classis Minecraft aesthetic. Or
    you can store geometric information, commonly referred to as "signed
    distances" or "hermite data," at each voxel in order to approximate smooth
    surfaces. We would like to support both schemes.
  - For cubic meshes, the fastest algorithm we know of to produce efficient
    meshes is coined as "Greedy Meshing" by the 0fps blog.
  - For smooth meshes, the most pervasive algorithm is Marching Cubes. However,
    we found the Naive Surface Nets algorithm to be simpler to implement and
    just as efficient, if not moreso.
  - We've also considered the Dual Contouring family of algorithms for smooth
    meshing. While they offer more control over the shape of a mesh, they are
    also more complex and thus expensive to compute. For now, we've decided not
    to pursue these algorithms, but we are open to any contributions in this
    area.
  - While 3D voxel data is required for meshes with arbitrary topologies, one
    can choose to constrain themselves to a simpler planar topology and reap
    performance benefits, both in terms of space and time. A surface with planar
    topology can be modeled with a 2-dimensional function commonly referred to
    as a "height map." While we could represent a height map with an `Array3`
    where one of the dimensions has size 1, it leads to awkward code. Thus, we
    generalized all of the core data types to work in both 2 and 3 dimensions,
    which gives us the `Array2` type, capable of cleanly representing a height
    map. We've also implemented a specialized meshing algorithm for height maps.
- **Accelerated Spatial Queries**
  - Our first voxel game prototypes utilized the ncollide3d crate and it's
    `DBVT` (dynamic bounding volume tree) structure for doing raycasting.
    Unforunately, storing an `AABB<f32>` for every voxel cost us 6 `f32`s or 24
    bytes per voxel. That simply doesn't scale. So as a replacement, we
    implemented the `Octree` and `OctreeDBVT` types. The `Octree` is essentially
    a hierarchical bitset, making it very memory efficient; it doesn't contain
    any voxel data, but it can tell you whether a `Point3i` is contained in the
    set. More importantly, it supports a visitor API that can be used for
    spatial queries like raycasting. Since the `Octree` has a limited size, we
    also provide the `OctreeDBVT`, which is essentially a `DBVT` which may
    contain an arbitrary number of `Octree`s.
  - `OctreeDBVT` requires taking on the `ncollide3d` dependency. We decided this
    was acceptable for the time being, since we don't have spare time to
    implement our own efficient `DBVT`.

## Roadmap

1. `Octree` from height map
2. procedurally generated heightmaps for terrain
    1. fractal noise
    2. hydraulic erosion
3. Bevy integration
    1. ECS systems for dynamic meshing
4. Rapier3d integration
5. level of detail
    1. stitching chunks on LoD boundaries
    2. geomorphing
6. GPU acceleration of core algorithms
7. SIMD variants of core data types
