//! This module contains octrees in various forms. The most fundamental of these is the [`OctreeSet`], which is used to
//! implement everything else. Because the `OctreeSet` has a bounded domain, it is extended to being unbounded by storing
//! multiple [`OctreeSet`]s in a hash map. This is known as the [`ChunkedOctreeSet`].
//!
//! While each octree is a set of `Point3i`s, they can also be used as sets of [`ChunkKey3`](crate::chunk::ChunkKey3)s. After
//! all, `ChunkKey`s are just `Point`s at a different scale. The [`OctreeChunkIndex`] satisfies this use case; it can be used to
//! index an unbounded set of chunks, which makes it a nice companion for the [`ChunkMap3`](crate::chunk::ChunkMap3). It can
//! also be used as a clipmap for level of detail.

pub mod chunk_index;
pub mod chunked_set;
pub mod clipmap;
pub mod set;

pub use chunk_index::*;
pub use chunked_set::*;
pub use clipmap::*;
pub use set::*;
