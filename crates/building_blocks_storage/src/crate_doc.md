Various types of storage and indexing for voxels in 2 or 3 dimensions.

The core storage types are:

- [`Array`](self::array): N-dimensional, single resolution, bounded,
  dense array
- [`ChunkTree`](self::chunk_tree): N-dimensional, multiple resolution,
  unbounded, sparse array
  - Backed by generic chunk storage, with [`SmallKeyHashMap`] or
    [`CompressibleChunkStorage`](self::chunk_tree::CompressibleChunkStorage) implementations
- [`ChunkDb`](self::database::ChunkDb): A persistent database for chunks backed by `sled`.

Then there are "meta" lattice maps that provide some extra utility:

- [`TransformMap`](self::transform_map::TransformMap): a wrapper of any kind of
  lattice map that performs an arbitrary transformation
- [`Func`](self::func::Func): some lattice map traits are implemented for
  closures (like SDFs)

If you need to store signed distance values in your voxels, consider using the
[`Sd8`](self::signed_distance::Sd8) and [`Sd16`](self::signed_distance::Sd16)
fixed-precision types which implement the
[`SignedDistance`](self::signed_distance::SignedDistance) trait required for
smooth meshing.
