# Design Philosophy and Architecture

## Principles

The design of Building Blocks is driven by a few guiding principles:

- **Real-Time Performance**
  - The primary use case for Building Blocks is using voxel technology within a
    video game. This means it needs to be fast. For example, we want to be able
    to generate meshes for millions of voxels per frame (16.6 ms).
  - Critical algorithms must be benchmarked with Criterion so we can guide
    optimization with evidence.
- **Composable APIs**
  - APIs are more powerful when they are generic. You will find many examples of
    generic APIs that require the input types to implement some traits.
  - This is most prevalent in the storage crate, where we desire all of the
    lattice map types to be accessible through the same core set of traits. This
    results in a very powerful function, `copy_extent`, which works with all
    implementations of the `ReadExtent` and `WriteExtent` traits.
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

## Architecture

![storage diagram](https://i.imgur.com/VPy3K36.png)

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
  - In most cases, the full extent of a voxel world will be only sparsely populated,
    and it is desirable to store only voxels close to "volume surfaces," whatever
    that may mean in a particular application. So we partition the lattice into
    chunks, each of which is an `Array3` of the same shape. Only chunks that capture
    some part of the surface are stored. The container for these chunks is called a
    `ChunkMap3`.
  - With both the `Array3` and `ChunkMap3` serving similar purposes, we've made
    them implement a common set of traits for data access. This includes random
    access, iteration, and copying, using the `Get*`, `ForEach*`, `ReadExtent`,
    and `WriteExtent` traits.
  - When you have large voxel worlds, it's not feasible to store a lot of unique
    data for every voxel. A common strategy is to have each voxel labeled with
    some "type" identifier. If you only want to use a single byte for each voxel's
    type ID, then you can have up to 256 types of voxels. Then each type can have a
    large amount of data associated with it in a "palette" container. But we still
    want to be able to use our common set of access traits to read the voxel
    type data. Thus, we have a type called `TransformMap` which implements those
    traits. `TransformMap` wraps some other lattice map, referred to as the
    "delegate," and any access will first go through the delegate, then be
    transformed by an arbitrary `Fn(T) -> S` chosen by the programmer. This
    transformation closure is where we can access the palette based on the voxel
    type provided by the delegate map.
  - Even with only a couple bytes per voxel, we can still use up lots of memory
    on large voxel maps. One simple way to save memory without changing the
    underlying array containers is to use compression on chunks. In order to avoid
    decompressing the same chunks many times, LRU caching is used so that the working
    set can stay decompressed. This functionality is encapsulated in the
    `CompressibleChunkStorage`.
  - To make chunk compression optional, we made `ChunkMap` take a new chunk storage type
    parameter that can implement `ChunkReadStorage` or `ChunkWriteStorage`. This
    makes it so `ChunkMap`s can work with any kind of backing storage, be it a
    simple hash map or something more complex.
  - Due to the requirement for level of detail, some hierarchical storage types
    were introduced. The `ChunkMap` and chunk storages were extended to support
    storing chunks at multiple levels of detail. There is also an `OctreeSet` which
    is a sparse, hierarchical set of `Point3i`s. This supports many kinds of traversal,
    so it can be used to efficiently visit large regions of chunk keys. Because the
    `OctreeSet` has a maximum size, the `OctreeChunkIndex` is a hash map of `OctreeSet`s
    where each set manages an extent that we call a "superchunk" (a multitude of chunks).
    Some algorithms, like clipmap traversal, have been implemented using these octrees.
- **Meshing**
  - There are many ways of generating meshes from voxel data. You can make each
    occupied voxel into a cube, which gives the classis Minecraft aesthetic. Or
    you can store geometric information, commonly referred to as "signed
    distances" or "hermite data," at each voxel in order to approximate smooth
    surfaces. We would like to support both schemes.
  - For cubic meshes, the fastest algorithm we know of to produce efficient
    meshes is coined as "Greedy Meshing" by the 0fps blog.
  - For smooth meshes, the most pervasive algorithm is Marching Cubes which is
    referred to as a "primal method." Marching Cubes places vertices on the
    edges of a voxel and constructs an independent mesh component for each
    voxel. There are also "dual methods," which place vertices on the interior
    of each voxel and connect them using a subset of the dual graph of the voxel
    lattice. We have found (Naive) Surface Nets, a dual method, to be simpler to
    implement, just as fast, and it produces meshes with fewer vertices at the
    same resolution as Marching Cubes.
  - Another dual method, "Dual Contouring of Hermite Data," is essentially the
    same as Surface Nets, but it optimizes quadratic error functions (QEFs) to
    place vertices in more accurate locations, which helps with reproducing sharp
    features. This algorithm is rather difficult to optimize, so we are consciously
    taking it off the table in the near term.
  - While 3D voxel data is required for meshes with arbitrary topologies, one
    can choose to constrain themselves to a simpler planar topology and reap
    performance benefits, both in terms of space and time. A surface with planar
    topology can be modeled with a 2-dimensional function commonly referred to
    as a "height map." While we could represent a height map with an `Array3`
    where one of the dimensions has size 1, it leads to awkward code. Thus, we
    generalized all of the core data types to work in both 2 and 3 dimensions,
    which gives us the `Array2` type, capable of cleanly representing a height
    map. We've also implemented a meshing algorithm for height maps.
  - For large maps, using a single lattice resolution for meshes will quickly
    eat up GPU resources. We must have a level of detail solution to solve this
    problem. This can be a complex issue, but for now we have settled on using
    multiresolution Surface Nets, which involves downsampling the lattice to
    conform to a clipmap structure. On LOD boundaries, we will need the higher
    resolution chunk to be aware of the faces where it borders a chunk of half
    resolution so that it can create the appropriate boundary mesh. This algorithm
    is essentially the same as dual contouring of an octree, except we do so
    on uniform grids for performance reasons.
