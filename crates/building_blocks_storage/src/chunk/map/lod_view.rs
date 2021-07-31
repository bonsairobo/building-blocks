use crate::{
    array::ArrayCopySrc,
    dev_prelude::{
        AmbientExtent, Chunk, ChunkKey, ChunkMap, ChunkMapBuilder, ChunkReadStorage,
        ChunkWriteStorage, FillExtent, ForEach, ForEachMut, ForEachMutPtr, Get, GetMut,
        GetMutUnchecked, GetRef, GetRefUnchecked, GetUnchecked, ReadExtent, WriteExtent,
    },
    multi_ptr::*,
};

use building_blocks_core::{point_traits::IntegerPoint, ExtentN, PointN};

use either::Either;
use std::ops::{Deref, DerefMut};

/// A view of a single level of detail in a `ChunkMap` for the unambiguous implementation of access traits.
pub struct ChunkMapLodView<Delegate> {
    pub delegate: Delegate,
    pub lod: u8,
}

impl<Delegate> ChunkMapLodView<Delegate> {
    #[inline]
    pub fn lod(&self) -> u8 {
        self.lod
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, Delegate, N, T, Bldr, Store> Get<PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: GetUnchecked<PointN<N>, Item = T>,
    Store: ChunkReadStorage<N, Bldr::Chunk>,
{
    type Item = T;

    #[inline]
    fn get(&self, p: PointN<N>) -> Self::Item {
        self.delegate.clone_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Bldr, Store, Ref> GetRef<'a, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: GetRefUnchecked<'a, PointN<N>, Item = Ref>,
    Store: 'a + ChunkReadStorage<N, Bldr::Chunk>,
    Ref: MultiRef<'a, Data = T>,
{
    type Item = Ref;

    #[inline]
    fn get_ref(&'a self, p: PointN<N>) -> Self::Item {
        self.delegate.get_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Bldr, Store, Mut> GetMut<'a, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: GetMutUnchecked<'a, PointN<N>, Item = Mut>,
    Store: 'a + ChunkWriteStorage<N, Bldr::Chunk>,
{
    type Item = Mut;

    #[inline]
    fn get_mut(&'a mut self, p: PointN<N>) -> Self::Item {
        self.delegate.get_mut_point(self.lod, p)
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

impl<Delegate, N, T, Bldr, Store> ForEach<N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: ForEach<N, PointN<N>, Item = T>,
    T: Clone,
    Store: ChunkReadStorage<N, Bldr::Chunk>,
{
    type Item = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.delegate
            .visit_chunks(self.lod, extent, |chunk| match chunk {
                Either::Left(chunk) => {
                    chunk.array().for_each(extent, |p, value| f(p, value));
                }
                Either::Right((chunk_extent, ambient)) => {
                    ambient.for_each(&extent.intersection(chunk_extent), |p, value| f(p, value))
                }
            });
    }
}

impl<Delegate, N, T, Bldr, Store, MutPtr> ForEachMutPtr<N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    Store: ChunkWriteStorage<N, Bldr::Chunk>,
{
    type Item = MutPtr;

    #[inline]
    unsafe fn for_each_mut_ptr(
        &mut self,
        extent: &ExtentN<N>,
        mut f: impl FnMut(PointN<N>, Self::Item),
    ) {
        self.delegate.visit_mut_chunks(self.lod, extent, |chunk| {
            chunk
                .array_mut()
                .for_each_mut_ptr(extent, |p, ptr| f(p, ptr))
        });
    }
}

impl<'a, Delegate, N, Mut, MutPtr> ForEachMut<'a, N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Self: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    MutPtr: IntoMultiMut<'a, MultiMut = Mut>,
{
    type Item = Mut;

    #[inline]
    fn for_each_mut(&'a mut self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        unsafe { self.for_each_mut_ptr(extent, |p, ptr| f(p, ptr.into_multi_mut())) }
    }
}

impl<Delegate, N, T, MutPtr> FillExtent<N> for ChunkMapLodView<Delegate>
where
    Self: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    MutPtr: MultiMutPtr<Data = T>,
    T: Clone,
{
    type Item = T;

    /// Fill all of `extent` with the same `value`.
    #[inline]
    fn fill_extent(&mut self, extent: &ExtentN<N>, value: T) {
        // PERF: write whole chunks using a fast path
        unsafe {
            self.for_each_mut_ptr(extent, |_p, ptr| ptr.write(value.clone()));
        }
    }
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

impl<'a, Delegate, N, T, Bldr, Store> ReadExtent<'a, N> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: 'a + ChunkMapBuilder<N, T>,
    T: 'a + Clone,
    Store: 'a + ChunkReadStorage<N, Bldr::Chunk>,
{
    type Src = ChunkCopySrc<N, T, &'a Bldr::Chunk>;
    type SrcIter = ChunkCopySrcIter<N, T, &'a Bldr::Chunk>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let chunk_iters = self
            .delegate
            .indexer
            .chunk_mins_for_extent(extent)
            .map(|chunk_min| {
                let chunk_extent = self.delegate.indexer.extent_for_chunk_with_min(chunk_min);
                let intersection = extent.intersection(&chunk_extent);

                (
                    intersection,
                    self.delegate
                        .get_chunk(ChunkKey::new(self.lod, chunk_min))
                        .map(|chunk| Either::Left(ArrayCopySrc(chunk)))
                        .unwrap_or_else(|| {
                            Either::Right(AmbientExtent::new(self.delegate.builder.ambient_value()))
                        }),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If `Array` supports writing from type Src, then so does ChunkMap.
impl<Delegate, N, T, Bldr, Store, Src> WriteExtent<N, Src> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Bldr: ChunkMapBuilder<N, T>,
    <Bldr::Chunk as Chunk>::Array: WriteExtent<N, Src>,
    Store: ChunkWriteStorage<N, Bldr::Chunk>,
    Src: Clone,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        self.delegate.visit_mut_chunks(self.lod, extent, |chunk| {
            chunk.array_mut().write_extent(extent, src.clone())
        });
    }
}

#[doc(hidden)]
pub type ChunkCopySrc<N, T, Ch> = Either<ArrayCopySrc<Ch>, AmbientExtent<N, T>>;
#[doc(hidden)]
pub type ChunkCopySrcIter<N, T, Ch> = std::vec::IntoIter<(ExtentN<N>, ChunkCopySrc<N, T, Ch>)>;
