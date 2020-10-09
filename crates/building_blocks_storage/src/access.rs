//! Traits defining different ways to access data from generic lattice maps.
//!
//! The fastest way to iterate over data in an Array is with a simple for loop over array indices,
//! we call them "stride"s:
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
//! let mut map = Array3::fill(extent, 0);
//!
//! for i in 0..extent.num_points() {
//!     // Use the `GetMut<Stride>` trait impl of the map.
//!     *map.get_mut(Stride(i)) = 1;
//! }
//! ```
//! But this may require understanding the array layout.
//!
//! Often, you only want to iterate over a sub-extent of the map. This can also be done at similar
//! speeds using the `ForEachRef` and `ForEachMut` traits:
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
//! # let mut map = Array3::fill(extent, 0);
//! let subextent = Extent3i::from_min_and_shape(PointN([1; 3]), PointN([98; 3]));
//! // Use the `ForEachMut<[i32; 3], Stride>` trait.
//! map.for_each_mut(&subextent, |_s: Stride, value: &mut i32| { *value = 2 });
//! ```
//! Arrays also implement `ForEach*<PointN<N>>` and `ForEach*<(PointN<N>, Stride)>`. `ChunkMap` and
//! `ChunkMapReader` only implement `ForEach*<PointN<N>>`, because it's ambiguous which chunk a
//! `Stride` would apply to.
//!
//! If you need to copy data between lattice maps, you should use the `copy_extent` function. Copies
//! can be done efficiently because the `ReadExtent` and `WriteExtent` traits allow lattice maps to
//! define how they would like to be written to or read from.
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(PointN([0; 3]), PointN([100; 3]));
//! # let mut map = Array3::fill(extent, 0);
//! # let subextent = Extent3i::from_min_and_shape(PointN([1; 3]), PointN([98; 3]));
//! // Create another map to copy to/from. We use a `ChunkMap`, but any map that implements
//! // `WriteExtent` can be a copy destination, and any map that implements `ReadExtent` can be a
//! // copy source.
//! let chunk_shape = PointN([16; 3]);
//! let ambient_value = 0;
//! let default_chunk_metadata = ();
//! let mut other_map = ChunkMap3::new(
//!     chunk_shape, ambient_value, default_chunk_metadata, FastLz4 { level: 10 }
//! );
//! copy_extent(&subextent, &map, &mut other_map);
//! let local_cache = LocalChunkCache::new();
//! let reader = ChunkMapReader3::new(&other_map, &local_cache);
//! copy_extent(&subextent, &reader, &mut map);
//!```

use building_blocks_core::ExtentN;

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

pub trait Get<L> {
    type Data;

    /// Get an owned value at `location`.
    fn get(&self, location: L) -> Self::Data;
}

pub trait GetRef<L> {
    type Data;

    /// Get a reference to the value at `location`.
    fn get_ref(&self, location: L) -> &Self::Data;
}

pub trait GetMut<L> {
    type Data;

    /// Get a mutable reference to the value at `location`.
    fn get_mut(&mut self, location: L) -> &mut Self::Data;
}

pub trait GetUnchecked<L> {
    type Data;

    /// Get the value at location without doing bounds checking.
    /// # Safety
    /// Don't access out of bounds.
    unsafe fn get_unchecked(&self, location: L) -> Self::Data;
}

pub trait GetUncheckedRef<L> {
    type Data;

    /// Get a reference to the value at location without doing bounds checking.
    /// # Safety
    /// Don't access out of bounds.
    unsafe fn get_unchecked_ref(&self, location: L) -> &Self::Data;
}

pub trait GetUncheckedMut<L> {
    type Data;

    /// Get a mutable reference to the value at location without doing bounds checking.
    /// # Safety
    /// Don't access out of bounds.
    unsafe fn get_unchecked_mut(&mut self, location: L) -> &mut Self::Data;
}

pub trait GetUncheckedRelease<L, T>: Get<L, Data = T> + GetUnchecked<L, Data = T> {
    #[inline]
    fn get_unchecked_release(&self, location: L) -> T {
        if cfg!(debug_assertions) {
            self.get(location)
        } else {
            unsafe { self.get_unchecked(location) }
        }
    }
}

impl<M, L, T> GetUncheckedRelease<L, T> for M where M: Get<L, Data = T> + GetUnchecked<L, Data = T> {}

/// A lattice map that supports getting without bounds checking only in release mode.
pub trait GetUncheckedRefRelease<L, T>: GetRef<L, Data = T> + GetUncheckedRef<L, Data = T> {
    #[inline]
    fn get_unchecked_ref_release(&self, location: L) -> &T {
        if cfg!(debug_assertions) {
            self.get_ref(location)
        } else {
            unsafe { self.get_unchecked_ref(location) }
        }
    }
}

impl<M, L, T> GetUncheckedRefRelease<L, T> for M where
    M: GetRef<L, Data = T> + GetUncheckedRef<L, Data = T>
{
}

/// A lattice map that supports getting without bounds checking only in release mode.
pub trait GetUncheckedMutRelease<L, T>: GetMut<L, Data = T> + GetUncheckedMut<L, Data = T> {
    #[inline]
    fn get_unchecked_mut_release(&mut self, location: L) -> &mut T {
        if cfg!(debug_assertions) {
            self.get_mut(location)
        } else {
            unsafe { self.get_unchecked_mut(location) }
        }
    }
}

impl<M, L, T> GetUncheckedMutRelease<L, T> for M where
    M: GetMut<L, Data = T> + GetUncheckedMut<L, Data = T>
{
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

pub trait ForEachRef<N, Coord> {
    type Data;

    fn for_each_ref(&self, extent: &ExtentN<N>, f: impl FnMut(Coord, &Self::Data));
}

pub trait ForEachMut<N, Coord> {
    type Data;

    fn for_each_mut(&mut self, extent: &ExtentN<N>, f: impl FnMut(Coord, &mut Self::Data));
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

/// Some lattice maps, like `ChunkedArray`, have nonlinear layouts. This means that, in order for a
/// writer to receive data efficiently, it must come as an iterator over multiple arrays.
pub trait ReadExtent<'a, N> {
    type Src: 'a;
    type SrcIter: Iterator<Item = (ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter;
}

pub trait WriteExtent<N, Src> {
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src);
}

/// Copy all points in `extent` from the `src` map to the `dst` map.
pub fn copy_extent<'a, N, Src: 'a, Ms, Md>(extent: &ExtentN<N>, src_map: &'a Ms, dst_map: &mut Md)
where
    Ms: ReadExtent<'a, N, Src = Src>,
    Md: WriteExtent<N, Src>,
{
    for (extent, extent_src) in src_map.read_extent(extent) {
        dst_map.write_extent(&extent, extent_src);
    }
}
