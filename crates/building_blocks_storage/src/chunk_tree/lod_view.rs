use crate::{
    array::ArrayCopySrc,
    dev_prelude::{
        AmbientExtent, ChunkKey, ChunkStorage, ChunkTree, ChunkTreeBuilder, FillExtent, ForEach,
        ForEachMut, ForEachMutPtr, Get, GetMut, GetMutUnchecked, GetRef, GetRefUnchecked,
        GetUnchecked, IterChunkKeys, ReadExtent, UserChunk, WriteExtent,
    },
    multi_ptr::*,
};

use building_blocks_core::{
    point_traits::{Bounded, IntegerPoint, LatticeOrder},
    ExtentN, PointN,
};

use either::Either;
use std::ops::{Deref, DerefMut};

/// A view of a single level of detail in a `ChunkTree` for the unambiguous implementation of access traits.
pub struct ChunkTreeLodView<Delegate> {
    pub delegate: Delegate,
    pub lod: u8,
}

impl<Delegate> ChunkTreeLodView<Delegate> {
    #[inline]
    pub fn lod(&self) -> u8 {
        self.lod
    }
}

impl<Delegate, N, T, Bldr, Store, Usr> ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    Store: ChunkStorage<N, Chunk = Usr>,
    PointN<N>: IntegerPoint,
{
    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be represented by an `AmbientExtent`.
    #[inline]
    pub fn visit_chunks(
        &self,
        extent: ExtentN<N>,
        mut visitor: impl FnMut(Either<&Usr, (&ExtentN<N>, AmbientExtent<N, T>)>),
    ) where
        T: Clone,
    {
        // PERF: we could traverse the octree to avoid using hashing to check for occupancy
        for chunk_min in self.delegate.indexer.chunk_mins_for_extent(&extent) {
            if let Some(chunk) = self.delegate.get_chunk(ChunkKey::new(self.lod, chunk_min)) {
                visitor(Either::Left(chunk))
            } else {
                let chunk_extent = self.delegate.indexer.extent_for_chunk_with_min(chunk_min);
                visitor(Either::Right((
                    &chunk_extent,
                    AmbientExtent::new(self.delegate.ambient_value.clone()),
                )))
            }
        }
    }
}

impl<'a, Delegate, N, T, Bldr, Store> ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    Store: ChunkStorage<N> + for<'r> IterChunkKeys<'r, N>,
    Bldr: ChunkTreeBuilder<N, T>,
{
    /// The smallest extent that bounds all chunks in this level of detail.
    pub fn bounding_extent(&self) -> Option<ExtentN<N>> {
        let root_lod = self.delegate.root_lod();

        let mut min_root_key = PointN::MAX;
        let mut max_root_key = PointN::MIN;
        self.delegate.visit_root_keys(|root_key| {
            min_root_key = root_key.minimum.meet(min_root_key);
            max_root_key = root_key.minimum.join(max_root_key);
        });

        // Find the minimal chunk key in the min root tree.
        let min_lod_key = self.min_key_recursive(ChunkKey::new(root_lod, min_root_key));
        // Find the maximal chunk key in the max root tree.
        let max_lod_key = self.max_key_recursive(ChunkKey::new(root_lod, max_root_key));

        let lub = max_lod_key + self.delegate.chunk_shape();

        ExtentN::from_min_and_lub(min_lod_key, lub).check_positive_shape()
    }

    fn min_key_recursive(&self, node_key: ChunkKey<N>) -> PointN<N> {
        if node_key.lod > self.lod {
            let mut min_key = PointN::MAX;
            self.delegate.visit_child_keys(node_key, |child_key, _| {
                min_key = min_key.meet(self.min_key_recursive(child_key));
            });
            min_key
        } else {
            node_key.minimum
        }
    }

    fn max_key_recursive(&self, node_key: ChunkKey<N>) -> PointN<N> {
        if node_key.lod > self.lod {
            let mut max_key = PointN::MIN;
            self.delegate.visit_child_keys(node_key, |child_key, _| {
                max_key = max_key.join(self.max_key_recursive(child_key));
            });
            max_key
        } else {
            node_key.minimum
        }
    }
}

impl<Delegate, N, T, Bldr, Store, Usr> ChunkTreeLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkTree<N, T, Bldr, Store>>,
    Store: ChunkStorage<N, Chunk = Usr>,
    PointN<N>: IntegerPoint,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
{
    /// Call `visitor` on all chunks that overlap `extent`. Vacant chunks will be created first with ambient value.
    #[inline]
    pub fn visit_mut_chunks(&mut self, extent: &ExtentN<N>, mut visitor: impl FnMut(&mut Usr)) {
        for chunk_min in self.delegate.indexer.chunk_mins_for_extent(extent) {
            visitor(
                self.delegate
                    .get_mut_chunk_or_insert_ambient(ChunkKey::new(self.lod, chunk_min)),
            );
        }
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, Delegate, N, T, Usr, Bldr, Store> Get<PointN<N>> for ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: GetUnchecked<PointN<N>, Item = T>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
    type Item = T;

    #[inline]
    fn get(&self, p: PointN<N>) -> Self::Item {
        self.delegate.clone_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Usr: 'a, Bldr: 'a, Store: 'a, Ref> GetRef<'a, PointN<N>>
    for ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: GetRefUnchecked<'a, PointN<N>, Item = Ref>,
    Store: ChunkStorage<N, Chunk = Usr>,
    Ref: MultiRef<'a, Data = T>,
{
    type Item = Ref;

    #[inline]
    fn get_ref(&'a self, p: PointN<N>) -> Self::Item {
        self.delegate.get_point(self.lod, p)
    }
}

impl<'a, Delegate, N, T: 'a, Usr: 'a, Bldr: 'a, Store: 'a, Mut> GetMut<'a, PointN<N>>
    for ChunkTreeLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    Usr: UserChunk,
    Usr::Array: GetMutUnchecked<'a, PointN<N>, Item = Mut>,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr>,
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

impl<Delegate, N, T, Usr, Bldr, Store> ForEach<N, PointN<N>> for ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    T: Clone,
    Usr: UserChunk,
    Usr::Array: ForEach<N, PointN<N>, Item = T>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
    type Item = T;

    #[inline]
    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        self.visit_chunks(*extent, |chunk| match chunk {
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
    for ChunkTreeLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    Usr: UserChunk,
    Usr::Array: ForEachMutPtr<N, PointN<N>, Item = MutPtr>,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr>,
{
    type Item = MutPtr;

    #[inline]
    unsafe fn for_each_mut_ptr(
        &mut self,
        extent: &ExtentN<N>,
        mut f: impl FnMut(PointN<N>, Self::Item),
    ) {
        self.visit_mut_chunks(extent, |chunk| {
            chunk
                .array_mut()
                .for_each_mut_ptr(extent, |p, ptr| f(p, ptr))
        });
    }
}

impl<'a, Delegate, N, Mut, MutPtr> ForEachMut<'a, N, PointN<N>> for ChunkTreeLodView<Delegate>
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

impl<Delegate, N, T, MutPtr> FillExtent<N> for ChunkTreeLodView<Delegate>
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
    for ChunkTreeLodView<Delegate>
where
    Delegate: Deref<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    T: Clone,
    Usr: UserChunk,
    Store: ChunkStorage<N, Chunk = Usr>,
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

// If `Array` supports writing from type Src, then so does ChunkTree.
impl<Delegate, N, T, Usr, Bldr, Store, Src> WriteExtent<N, Src> for ChunkTreeLodView<Delegate>
where
    Delegate: DerefMut<Target = ChunkTree<N, T, Bldr, Store>>,
    PointN<N>: IntegerPoint,
    Usr: UserChunk,
    Usr::Array: WriteExtent<N, Src>,
    Bldr: ChunkTreeBuilder<N, T, Chunk = Usr>,
    Store: ChunkStorage<N, Chunk = Usr>,
    Src: Clone,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src) {
        self.visit_mut_chunks(extent, |chunk| {
            chunk.array_mut().write_extent(extent, src.clone())
        });
    }
}

#[doc(hidden)]
pub type ChunkCopySrc<N, T, Ch> = Either<ArrayCopySrc<Ch>, AmbientExtent<N, T>>;
#[doc(hidden)]
pub type ChunkCopySrcIter<N, T, Ch> = std::vec::IntoIter<(ExtentN<N>, ChunkCopySrc<N, T, Ch>)>;
