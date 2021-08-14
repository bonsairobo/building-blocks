use crate::{
    array::ArrayCopySrc,
    chunk::ChunkNode,
    dev_prelude::{
        AmbientExtent, ChunkKey, ChunkMap, ChunkMapBuilder, ChunkStorage, FillExtent, ForEach,
        ForEachMut, ForEachMutPtr, Get, GetMut, GetMutUnchecked, GetRef, GetRefUnchecked,
        GetUnchecked, ReadExtent, UserChunk, WriteExtent,
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

impl<'a, Delegate, N, T, Usr, Bldr, Store> Get<PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: GetUnchecked<PointN<N>, Item = T>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    type Item = T;

    #[inline]
    fn get(&self, p: PointN<N>) -> Self::Item {
        self.delegate.clone_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Usr: 'a, Bldr: 'a, Store: 'a, Ref> GetRef<'a, PointN<N>>
    for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: GetRefUnchecked<'a, PointN<N>, Item = Ref>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
    Ref: MultiRef<'a, Data = T>,
{
    type Item = Ref;

    #[inline]
    fn get_ref(&'a self, p: PointN<N>) -> Self::Item {
        self.delegate.get_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Usr: 'a, Bldr: 'a, Store: 'a, Mut> GetMut<'a, PointN<N>>
    for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Usr::Array: GetMutUnchecked<'a, PointN<N>, Item = Mut>,
    Bldr: ChunkMapBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
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

impl<Delegate, N, T, Usr, Bldr, Store> ForEach<N, PointN<N>> for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: ForEach<N, PointN<N>, Item = T>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    type Item = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.delegate
            .visit_chunks(self.lod, *extent, |chunk| match chunk {
                Either::Left(chunk) => {
                    chunk.array().for_each(extent, |p, value| f(p, value));
                }
                Either::Right((chunk_extent, ambient)) => {
                    ambient.for_each(&extent.intersection(chunk_extent), |p, value| f(p, value))
                }
            });
    }
}

impl<Delegate, N, T, Usr, Bldr, Store, MutPtr> ForEachMutPtr<N, PointN<N>>
    for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Usr::Array: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    Bldr: ChunkMapBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
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

impl<'a, Delegate, N, T: 'a, Usr: 'a, Bldr: 'a, Store: 'a> ReadExtent<'a, N>
    for ChunkMapLodView<Delegate>
where
    Delegate: Deref<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Usr: UserChunk,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
{
    type Src = ChunkCopySrc<N, T, &'a Usr>;
    type SrcIter = ChunkCopySrcIter<N, T, &'a Usr>;

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
                            Either::Right(AmbientExtent::new(self.delegate.ambient_value.clone()))
                        }),
                )
            })
            .collect::<Vec<_>>();

        chunk_iters.into_iter()
    }
}

// If `Array` supports writing from type Src, then so does ChunkMap.
impl<Delegate, N, T, Usr, Bldr, Store, Src> WriteExtent<N, Src> for ChunkMapLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkMap<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint<N>,
    Usr: UserChunk,
    Usr::Array: WriteExtent<N, Src>,
    Bldr: ChunkMapBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = ChunkNode<Usr>>,
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
