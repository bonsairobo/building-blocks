//! Traits defining different ways to access data from generic lattice maps.
//!
//! # Strided Iteration
//!
//! The fastest way to iterate over data in an Array is with a simple for loop over array indices, we call them "stride"s:
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! let mut map = Array3::fill(extent, 0);
//!
//! for i in 0..extent.num_points() {
//!     // Use the `GetMut<Stride>` trait impl of the map.
//!     *map.get_mut(Stride(i)) = 1;
//! }
//! ```
//! But this may require understanding the array layout.
//!
//! # `ForEach` over Extent
//!
//! Often, you only want to iterate over a sub-extent of the map. This can also be done at similar speeds using the `ForEach`
//! and `ForEachMut` traits:
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! # let mut map = Array3::fill(extent, 0);
//! let subextent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(98));
//! // Use the `ForEachMut<[i32; 3], Stride>` trait.
//! map.for_each_mut(&subextent, |_s: Stride, value| { *value = 2 });
//! ```
//! Arrays also implement `ForEach*<PointN<N>>` and `ForEach*<(PointN<N>, Stride)>`. `ChunkMap` only implements
//! `ForEach*<PointN<N>>`, because it's ambiguous which chunk a `Stride` would apply to.
//!
//! # Copy an Extent
//!
//! If you need to copy data between lattice maps, you should use the `copy_extent` function. Copies can be done efficiently
//! because the `ReadExtent` and `WriteExtent` traits allow lattice maps to define how they would like to be written to or read
//! from.
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(100));
//! # let mut map = Array3::fill(extent, 0);
//! # let subextent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(98));
//! // Create another map to copy to/from. We use a `ChunkHashMap`, but any map that implements
//! // `WriteExtent` can be a copy destination, and any map that implements `ReadExtent` can be a
//! // copy source.
//! let builder = ChunkMapBuilder { chunk_shape: Point3i::fill(16), ambient_value: 0, default_chunk_metadata: () };
//! let mut other_map = builder.build_with_hash_map_storage();
//! copy_extent(&subextent, &map, &mut other_map);
//! copy_extent(&subextent, &other_map, &mut map);
//!
//! // You can even copy from a `Fn(Point3i) -> T`.
//! copy_extent(&subextent, &|p: Point3i| p.x(), &mut map);
//!```

use building_blocks_core::ExtentN;

// TODO: GATs should make it possible to collapse these traits for T, &T, and &mut T.

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

    /// Get an immutable reference to the value at `location`.
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

    /// Get an immutable reference to the value at `location` without doing bounds checking.
    /// # Safety
    /// Don't access out of bounds.
    unsafe fn get_unchecked_ref(&self, location: L) -> &Self::Data;
}

pub trait GetUncheckedMut<L> {
    type Data;

    /// Get a mutable reference to the value at `location` without doing bounds checking.
    /// # Safety
    /// Don't access out of bounds.
    unsafe fn get_unchecked_mut(&mut self, location: L) -> &mut Self::Data;
}

pub trait GetUncheckedRelease<L, T>: Get<L, Data = T> + GetUnchecked<L, Data = T> {
    /// Get the value at location. Skips bounds checking in release mode.
    /// # Safety
    /// Don't access out of bounds.
    #[inline]
    fn get_unchecked_release(&self, location: L) -> T {
        if cfg!(debug_assertions) {
            self.get(location)
        } else {
            unsafe { self.get_unchecked(location) }
        }
    }
}

impl<Map, L, T> GetUncheckedRelease<L, T> for Map where
    Map: Get<L, Data = T> + GetUnchecked<L, Data = T>
{
}

pub trait GetUncheckedRefRelease<L, T>: GetRef<L, Data = T> + GetUncheckedRef<L, Data = T> {
    /// Get an immutable reference to the value at `location`. Skips bounds checking in release mode.
    /// # Safety
    /// Don't access out of bounds.
    #[inline]
    fn get_unchecked_ref_release(&self, location: L) -> &T {
        if cfg!(debug_assertions) {
            self.get_ref(location)
        } else {
            unsafe { self.get_unchecked_ref(location) }
        }
    }
}

impl<Map, L, T> GetUncheckedRefRelease<L, T> for Map where
    Map: GetRef<L, Data = T> + GetUncheckedRef<L, Data = T>
{
}

pub trait GetUncheckedMutRelease<L, T>: GetMut<L, Data = T> + GetUncheckedMut<L, Data = T> {
    /// Get an mutable reference to the value at `location`. Skips bounds checking in release mode.
    /// # Safety
    /// Don't access out of bounds.
    #[inline]
    fn get_unchecked_mut_release(&mut self, location: L) -> &mut T {
        if cfg!(debug_assertions) {
            self.get_mut(location)
        } else {
            unsafe { self.get_unchecked_mut(location) }
        }
    }
}

impl<Map, L, T> GetUncheckedMutRelease<L, T> for Map where
    Map: GetMut<L, Data = T> + GetUncheckedMut<L, Data = T>
{
}

// We need this macro because doing a blanket impl causes conflicts due to Rust's orphan rules.
macro_rules! impl_get_via_get_ref_and_clone {
    ($map:ty, $($type_params:ident),*) => {
        impl<L, $($type_params),*> Get<L> for $map
        where
            Self: GetRef<L>,
            <Self as GetRef<L>>::Data: Clone,
        {
            type Data = <Self as GetRef<L>>::Data;
            #[inline]
            fn get(&self, location: L) -> Self::Data {
                self.get_ref(location).clone()
            }
        }
        impl<L, $($type_params),*> GetUnchecked<L> for $map
        where
            Self: GetUncheckedRef<L>,
            <Self as GetUncheckedRef<L>>::Data: Clone,
        {
            type Data = <Self as GetUncheckedRef<L>>::Data;
            #[inline]
            unsafe fn get_unchecked(&self, location: L) -> Self::Data {
                self.get_unchecked_ref(location).clone()
            }
        }
    };
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

pub trait ForEach<N, Coord> {
    type Item;

    fn for_each(&self, extent: &ExtentN<N>, f: impl FnMut(Coord, Self::Item));
}

pub trait ForEachMut<'a, N, Coord> {
    // TODO: use GAT to remove unsafe lifetime workaround in impls
    type Item;

    fn for_each_mut(&'a mut self, extent: &ExtentN<N>, f: impl FnMut(Coord, Self::Item));
}

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

/// A trait to facilitate the generic implementation of `copy_extent`.
///
/// Some lattice maps, like `ChunkMap`, have nonlinear layouts. This means that, in order for a writer to receive data
/// efficiently, it must come as an iterator over multiple extents.
pub trait ReadExtent<'a, N> {
    type Src;
    type SrcIter: Iterator<Item = (ExtentN<N>, Self::Src)>;

    /// `SrcIter` must return extents that are subsets of `extent`.
    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter;
}

/// A trait to facilitate the generic implementation of `copy_extent`.
pub trait WriteExtent<N, Src> {
    fn write_extent(&mut self, extent: &ExtentN<N>, src: Src);
}

/// Copy all points in `extent` from the `src` map to the `dst` map.
pub fn copy_extent<'a, N, Src, Ms, Md>(extent: &ExtentN<N>, src_map: &'a Ms, dst_map: &mut Md)
where
    Ms: ReadExtent<'a, N, Src = Src>,
    Md: WriteExtent<N, Src>,
{
    for (sub_extent, extent_src) in src_map.read_extent(extent) {
        dst_map.write_extent(&sub_extent, extent_src);
    }
}
