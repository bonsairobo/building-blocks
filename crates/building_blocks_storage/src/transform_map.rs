//! A lattice map that overlays a transformation on top of a delegate lattice map.
//!
//! As an example use case, say you have a large lattice map that can store various types of voxels, and each type of voxel has
//! some associated data. If that data is even moderately sized, it could take up a lot of space by storing copies at every
//! point of the lattice.
//!
//! Instead, you can store that data in a "palette" array, and store indices into that array as your voxel data.
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! struct BigData([u8; 9001]);
//!
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
//! let mut index_map = Array3x1::fill(extent, 0u8);
//! *index_map.get_mut(PointN([0, 0, 1])) = 1;
//!
//! let palette = vec![BigData([1; 9001]), BigData([2; 9001])];
//! let lookup = |i: u8| palette[i as usize].0[0];
//! let big_data_map = TransformMap::new(&index_map, &lookup);
//!
//! assert_eq!(big_data_map.get(PointN([0, 0, 0])), palette[0].0[0]);
//! assert_eq!(big_data_map.get(PointN([0, 0, 1])), palette[1].0[0]);
//! ```
//!
//! `TransformMap` also gives us an efficient way of applying transforms to array data during a copy:
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
//! let src = Array3x1::fill(extent, 0);
//! let chunk_shape = Point3i::fill(4);
//! let builder = ChunkMapBuilder3x1::new(chunk_shape, 0);
//! let mut dst = builder.build_with_hash_map_storage();
//! let tfm = TransformMap::new(&src, |value: i32| value + 1);
//! copy_extent(&extent, &tfm, &mut dst);
//! ```

use crate::{
    AmbientExtent, Array, ArrayCopySrc, ChunkCopySrc, ChunkCopySrcIter, ChunkMap, ChunkMapBuilder,
    ForEach, Get, IndexedArray, ReadExtent,
};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};

/// A lattice map that delegates look-ups to a different lattice map, then transforms the result
/// using some `Fn(In) -> Out`.
pub struct TransformMap<'a, Delegate, F> {
    delegate: &'a Delegate,
    transform: F,
}

impl<'a, Delegate, F> Clone for TransformMap<'a, Delegate, F>
where
    F: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.delegate, self.transform.clone())
    }
}

impl<'a, Delegate, F> Copy for TransformMap<'a, Delegate, F> where F: Copy {}

impl<'a, Delegate, F> TransformMap<'a, Delegate, F> {
    #[inline]
    pub fn new(delegate: &'a Delegate, transform: F) -> Self {
        Self {
            delegate,
            transform,
        }
    }
}

impl<'a, Delegate, F, In, Out, Coord> Get<Coord> for TransformMap<'a, Delegate, F>
where
    F: Fn(In) -> Out,
    Delegate: Get<Coord, Item = In>,
{
    type Item = Out;

    #[inline]
    fn get(&self, c: Coord) -> Self::Item {
        (self.transform)(self.delegate.get(c))
    }
}

impl<'a, N, Delegate, F, In, Out, Coord> ForEach<N, Coord> for TransformMap<'a, Delegate, F>
where
    F: Fn(In) -> Out,
    Delegate: ForEach<N, Coord, Item = In>,
{
    type Item = Out;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(Coord, Self::Item)) {
        self.delegate
            .for_each(extent, |c, t| f(c, (self.transform)(t)))
    }
}

impl<'a, N, Delegate, F> IndexedArray<N> for TransformMap<'a, Delegate, F>
where
    Delegate: IndexedArray<N>,
{
    type Indexer = Delegate::Indexer;

    #[inline]
    fn extent(&self) -> &ExtentN<N> {
        self.delegate.extent()
    }
}

// TODO: try to make a generic ReadExtent impl, it's hard because we need a way to define the src types as a function of the
// delegate src types (kinda hints at a monad or HKT)

impl<'a, N, Chan, F> ReadExtent<'a, N> for TransformMap<'a, Array<N, Chan>, F>
where
    Self: IndexedArray<N> + Clone,
    PointN<N>: IntegerPoint<N>,
{
    type Src = ArrayCopySrc<Self>;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let in_bounds_extent = self.extent().intersection(extent);

        once((in_bounds_extent, ArrayCopySrc(self.clone())))
    }
}

impl<'a, N, F, In, Out, B, Store> ReadExtent<'a, N>
    for TransformMap<'a, ChunkMap<N, In, B, Store>, F>
where
    F: Copy + Fn(In) -> Out,
    In: 'a,
    B: ChunkMapBuilder<N, In>,
    ChunkMap<N, In, B, Store>: ReadExtent<
        'a,
        N,
        Src = ChunkCopySrc<N, In, &'a B::Chunk>,
        SrcIter = ChunkCopySrcIter<N, In, &'a B::Chunk>,
    >,
{
    type Src = TransformChunkCopySrc<'a, N, F, Out, B::Chunk>;
    type SrcIter = TransformChunkCopySrcIter<'a, N, F, In, B::Chunk>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        TransformChunkCopySrcIter {
            chunk_iter: self.delegate.read_extent(extent),
            transform: self.transform,
        }
    }
}

#[doc(hidden)]
pub type TransformChunkCopySrc<'a, N, F, Out, Ch> = ChunkCopySrc<N, Out, TransformMap<'a, Ch, F>>;

#[doc(hidden)]
pub struct TransformChunkCopySrcIter<'a, N, F, In, Ch> {
    chunk_iter: ChunkCopySrcIter<N, In, &'a Ch>,
    transform: F,
}

impl<'a, N, F, In, Out, Ch> Iterator for TransformChunkCopySrcIter<'a, N, F, In, Ch>
where
    N: 'a,
    F: Copy + Fn(In) -> Out,
    In: 'a,
    Ch: 'a,
{
    type Item = (ExtentN<N>, TransformChunkCopySrc<'a, N, F, Out, Ch>);

    fn next(&mut self) -> Option<Self::Item> {
        self.chunk_iter.next().map(|(extent, chunk_src)| {
            (
                extent,
                chunk_src
                    .map_left(|array_src| {
                        ArrayCopySrc(TransformMap::new(array_src.0, self.transform))
                    })
                    .map_right(|ambient| AmbientExtent::new((self.transform)(ambient.value))),
            )
        })
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    const INT_BUILDER: ChunkMapBuilder3x1<i32> = ChunkMapBuilder3x1::new(PointN([4; 3]), 0);
    const FLOAT_BUILDER: ChunkMapBuilder3x1<f32> = ChunkMapBuilder3x1::new(PointN([4; 3]), 0.0);

    #[test]
    fn transform_accessors() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let inner_map: Array3x1<usize> = Array3x1::fill(extent, 0usize);

        let palette = vec![1, 2, 3];
        let outer_map = TransformMap::new(&inner_map, |i: usize| palette[i]);

        assert_eq!(outer_map.get(Point3i::ZERO), 1);

        outer_map.for_each(&extent, |_s: Stride, value| {
            assert_eq!(value, 1);
        });
        outer_map.for_each(&extent, |_p: Point3i, value| {
            assert_eq!(value, 1);
        });
        outer_map.for_each(&extent, |_ps: (Point3i, Stride), value| {
            assert_eq!(value, 1);
        });

        let outer_map = TransformMap::new(&inner_map, |i: usize| palette[i]);
        assert_eq!(outer_map.get(Point3i::ZERO), 1);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn copy_from_transformed_array() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let src = Array3x1::fill(extent, 0);
        let mut dst: ChunkHashMap3x1<f32> = FLOAT_BUILDER.build_with_hash_map_storage();
        let tfm = TransformMap::new(&src, |value: i32| value as f32 + 1.0);
        copy_extent(&extent, &tfm, &mut dst);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn copy_from_transformed_chunk_map_reader() {
        let src_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(16));
        let src_array = Array3x1::fill(src_extent, 1);
        let mut src = INT_BUILDER.build_with_hash_map_storage();
        copy_extent(&src_extent, &src_array, &mut src);

        let tfm = TransformMap::new(&src, |value: i32| value + 1);

        let dst_extent = Extent3i::from_min_and_shape(Point3i::fill(-16), Point3i::fill(32));
        let mut dst = INT_BUILDER.build_with_hash_map_storage();
        copy_extent(&dst_extent, &tfm, &mut dst);
    }
}
