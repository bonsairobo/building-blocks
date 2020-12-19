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
//! let extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
//! let mut index_map = Array3::fill(extent, 0u8);
//! *index_map.get_mut(&PointN([0, 0, 1])) = 1;
//!
//! let palette = vec![BigData([1; 9001]), BigData([2; 9001])];
//! let lookup = |i: u8| palette[i as usize].0[0];
//! let big_data_map = TransformMap::new(&index_map, &lookup);
//!
//! assert_eq!(big_data_map.get(&PointN([0, 0, 0])), palette[0].0[0]);
//! assert_eq!(big_data_map.get(&PointN([0, 0, 1])), palette[1].0[0]);
//! ```
//!
//! `TransformMap` also gives us an efficient way of applying transforms to array data during a copy:
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
//! let src = Array3::fill(extent, 0);
//! let builder = ChunkMapBuilder { chunk_shape: PointN([4; 3]), ambient_value: 0, default_chunk_metadata: () };
//! let mut dst = builder.build_with_hash_map_storage();
//! let tfm = TransformMap::new(&src, &|value: i32| value + 1);
//! copy_extent(&extent, &tfm, &mut dst);
//! ```

use crate::{
    access::GetUnchecked,
    array::{Array, ArrayCopySrc},
    chunk_map::{AmbientExtent, ArrayChunkCopySrc, ArrayChunkCopySrcIter, ChunkCopySrc, ChunkMap},
    ArrayN, ForEach, Get, ReadExtent,
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
        Self {
            delegate: self.delegate,
            transform: self.transform.clone(),
        }
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
    Delegate: Get<Coord, Data = In>,
{
    type Data = Out;

    #[inline]
    fn get(&self, c: Coord) -> Self::Data {
        (self.transform)(self.delegate.get(c))
    }
}

impl<'a, Delegate, F, In, Out, Coord> GetUnchecked<Coord> for TransformMap<'a, Delegate, F>
where
    F: Fn(In) -> Out,
    Delegate: GetUnchecked<Coord, Data = In>,
{
    type Data = Out;

    #[inline]
    unsafe fn get_unchecked(&self, c: Coord) -> Self::Data {
        (self.transform)(self.delegate.get_unchecked(c))
    }
}

impl<'a, Meta, F, In, Out, N, Coord> ForEach<N, Coord> for TransformMap<'a, Meta, F>
where
    F: Fn(In) -> Out,
    Meta: ForEach<N, Coord, Data = In>,
{
    type Data = Out;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(Coord, Self::Data)) {
        self.delegate
            .for_each(extent, |c, t| f(c, (self.transform)(t)))
    }
}

impl<'a, Delegate, F, N> Array<N> for TransformMap<'a, Delegate, F>
where
    Delegate: Array<N>,
{
    type Indexer = Delegate::Indexer;

    #[inline]
    fn extent(&self) -> &ExtentN<N> {
        self.delegate.extent()
    }
}

// TODO: try to make a generic ReadExtent impl, it's hard because we need a way to define the src
// types as a function of the delegate src types (kinda hints at a monad or HKT)

impl<'a, F, In, Out, N> ReadExtent<'a, N> for TransformMap<'a, ArrayN<N, In>, F>
where
    Self: Array<N> + Copy,
    F: 'a + Fn(In) -> Out,
    PointN<N>: IntegerPoint<N>,
{
    type Src = ArrayCopySrc<Self>;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let in_bounds_extent = self.extent().intersection(extent);

        once((in_bounds_extent, ArrayCopySrc(*self)))
    }
}

// Don't worry... just believe in the types and they will show you the way.
impl<'a, F, In, Out, N, Meta, Store> ReadExtent<'a, N>
    for TransformMap<'a, ChunkMap<N, In, Meta, Store>, F>
where
    ChunkMap<N, In, Meta, Store>: ReadExtent<
        'a,
        N,
        Src = ArrayChunkCopySrc<'a, N, In>,
        SrcIter = ArrayChunkCopySrcIter<'a, N, In>,
    >,
    F: 'a + Copy + Fn(In) -> Out,
    In: Copy,
    Out: 'a,
    Meta: Clone,
{
    type Src = TransformChunkCopySrc<'a, F, In, Out, N>;
    type SrcIter = TransformChunkCopySrcIter<'a, F, In, Out, N>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        TransformChunkCopySrcIter {
            chunk_iter: self.delegate.read_extent(extent),
            transform: self.transform,
        }
    }
}

#[doc(hidden)]
pub type TransformChunkCopySrc<'a, F, In, Out, N> =
    ChunkCopySrc<TransformMap<'a, ArrayN<N, In>, F>, N, Out>;

#[doc(hidden)]
pub struct TransformChunkCopySrcIter<'a, F, In, Out, N>
where
    F: Fn(In) -> Out,
{
    chunk_iter: ArrayChunkCopySrcIter<'a, N, In>,
    transform: F,
}

impl<'a, F, In, Out, N> Iterator for TransformChunkCopySrcIter<'a, F, In, Out, N>
where
    F: Copy + Fn(In) -> Out,
{
    type Item = (ExtentN<N>, TransformChunkCopySrc<'a, F, In, Out, N>);

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

    const BUILDER: ChunkMapBuilder3<i32> = ChunkMapBuilder {
        chunk_shape: PointN([4; 3]),
        ambient_value: 0,
        default_chunk_metadata: (),
    };

    #[test]
    fn transform_accessors() {
        let extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let inner_map: Array3<usize> = Array3::fill(extent, 0usize);

        let palette = vec![1, 2, 3];
        let outer_map = TransformMap::new(&inner_map, |i: usize| palette[i]);

        assert_eq!(outer_map.get(&PointN([0; 3])), 1);

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
        assert_eq!(outer_map.get(&PointN([0; 3])), 1);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn copy_from_transformed_array() {
        let extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let src = Array3::fill(extent, 0);
        let mut dst = BUILDER.build_with_hash_map_storage();
        let tfm = TransformMap::new(&src, |value: i32| value + 1);
        copy_extent(&extent, &tfm, &mut dst);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn copy_from_transformed_chunk_map_reader() {
        let src_extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([16; 3]));
        let src_array = Array3::fill(src_extent, 1);
        let mut src = BUILDER.build_with_hash_map_storage();
        copy_extent(&src_extent, &src_array, &mut src);

        let tfm = TransformMap::new(&src, |value: i32| value + 1);

        let dst_extent = Extent3::from_min_and_shape(PointN([-16; 3]), PointN([32; 3]));
        let mut dst = BUILDER.build_with_hash_map_storage();
        copy_extent(&dst_extent, &tfm, &mut dst);
    }
}
