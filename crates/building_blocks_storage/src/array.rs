//! N-dimensional arrays, where N is 2 or 3.
//!
//! The domains of all arrays are located within an ambient space, a signed integer lattice where the elements are `Point2i` or
//! `Point3i`. This means they contain data at exactly the set of points in an `ExtentN`, and no more.
//!
//! # Indexing
//!
//! You can index an array with 3 kinds of coordinates, with [`Get`](crate::access_traits) traits:
//!   - `Get*<Stride>`: flat array offset
//!   - `Get*<&LocalN>`: N-dimensional point in extent-local coordinates (i.e. min = `[0, 0, 0]`)
//!   - `Get*<PointN>`: N-dimensional point in global (ambient) coordinates
//!
//! Indexing assumes that the coordinates are in-bounds of the array, panicking otherwise.
//!
//! # Iteration
//!
//! Arrays also support fast iteration over extents with `ForEach*` trait impls. These methods will only iterate over the
//! section of the extent which is in-bounds of the array, so it's impossible to index out of bounds.
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let array_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(64));
//! let mut array = Array3x1::fill(array_extent, 0);
//!
//! // Write all points in the extent to the same value.
//! let write_extent = Extent3i::from_min_and_lub(Point3i::fill(10), Point3i::fill(20));
//! array.for_each_mut(&write_extent, |_: (), value| *value = 1);
//!
//! // Only the points in the extent should have been written.
//! array.for_each(array.extent(), |p: Point3i, value|
//!     if write_extent.contains(p) {
//!         assert_eq!(value, 1);
//!     } else {
//!         assert_eq!(value, 0);
//!     }
//! );
//! ```
//!
//! # Strides
//!
//! Since `Stride` lookups are fast and linear, they are ideal for kernel-based algorithms (like edge/surface detection). Use
//! the `ForEach*<N, Stride>` traits to iterate over an extent and use the linearity of `Stride` to access adjacent points.
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(64));
//! // Use a more interesting data set, just to show off this constructor.
//! let mut array = Array3x1::fill_with(extent, |p| if p.x() % 2 == 0 { 1 } else { 0 });
//!
//! let subextent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(62));
//!
//! // Some of these offsets include negative coordinates, which would underflow when translated
//! // into an unsigned index. That's OK though, because Stride is intended to be used with modular
//! // arithmetic.
//! let offsets = Local::localize_points(&Point3i::von_neumann_offsets());
//! let mut neighbor_strides = [Stride(0); 6];
//! array.strides_from_local_points(&offsets, &mut neighbor_strides);
//!
//! // Sum up the values in the Von Neumann neighborhood of each point, acting as a sort of blur
//! // filter.
//! array.for_each(&subextent, |stride: Stride, value| {
//!     let mut neighborhood_sum = value;
//!     for offset in neighbor_strides.iter() {
//!         let adjacent_value = array.get(stride + *offset);
//!         neighborhood_sum += adjacent_value;
//!     }
//! });
//! ```
//! This means you keep the performance of simple array indexing, as opposed to indexing with a `Point3i`, which requires 2
//! multiplications to convert to a `Stride`. You'd be surprised how important this difference can be in tight loops.
//!
//! # Storage
//!
//! By default, `ArrayNx1` uses a `Vec` to store elements. But any type that implements `Deref<Target = [T]>` or
//! `DerefMut<Target = [T]>` should be usable. This means you can construct an array with most pointer types.
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(64));
//! // Borrow `array`'s values for the lifetime of `other_array`.
//! let array = Array3x1::fill(extent, 1);
//! let other_array = Array3x1::new_one_channel(extent, array.channels().store().as_slice());
//! assert_eq!(other_array.get(Stride(0)), 1);
//!
//! // A stack-allocated array.
//! let mut data = [1; 64 * 64 * 64];
//! let mut stack_array = Array3x1::new_one_channel(extent, &mut data[..]);
//! *stack_array.get_mut(Stride(0)) = 2;
//! assert_eq!(data[0], 2);
//!
//! // A boxed array.
//! let data: Box<[u32]> = Box::new([1; 64 * 64 * 64]); // must forget the size
//! let box_array = Array3x1::new_one_channel(extent, data);
//! box_array.for_each(&extent, |p: Point3i, value| assert_eq!(value, 1));
//! ```

mod channel;
mod compression;
mod coords;
#[macro_use]
mod for_each;
mod indexer;

#[cfg(feature = "dot_vox")]
mod dot_vox_conversions;
#[cfg(feature = "image")]
mod image_conversions;

pub use channel::*;
pub use compression::*;
pub use coords::*;
pub use for_each::*;
pub use indexer::*;

use crate::{
    ChunkCopySrc, ForEach, ForEachMut, Get, GetMut, GetRef, IntoRawBytes, ReadExtent, TransformMap,
    WriteExtent,
};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};
use core::mem::MaybeUninit;
use core::ops::{Add, Deref, DerefMut};
use either::Either;
use serde::{Deserialize, Serialize};

/// A map from lattice location `PointN<N>` to data `T`, stored as a flat array.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Array<N, Chan> {
    channels: Chan,
    extent: ExtentN<N>,
}

/// An N-dimensional, 1-channel `Array`.
pub type ArrayNx1<N, T, Store = Vec<T>> = Array<N, Channel<T, Store>>;

/// A 2-dimensional, 1-channel `Array`.
pub type Array2x1<T, Store = Vec<T>> = ArrayNx1<[i32; 2], T, Store>;
/// A 3-dimensional, 1-channel `Array`.
pub type Array3x1<T, Store = Vec<T>> = ArrayNx1<[i32; 3], T, Store>;

impl<N, Chan> Array<N, Chan> {
    /// Create a new `ArrayNx1` directly from the extent and values. This asserts that the number of points in the extent matches
    /// the length of the values `Vec`.
    pub fn new(extent: ExtentN<N>, channels: Chan) -> Self {
        Self { extent, channels }
    }

    /// Moves the raw extent and values storage out of `self`.
    #[inline]
    pub fn into_parts(self) -> (ExtentN<N>, Chan) {
        (self.extent, self.channels)
    }

    #[inline]
    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }

    #[inline]
    pub fn channels(&self) -> &Chan {
        &self.channels
    }

    #[inline]
    pub fn channels_mut(&mut self) -> &mut Chan {
        &mut self.channels
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: Deref<Target = [T]>,
{
    pub fn new_one_channel(extent: ExtentN<N>, values: Store) -> Self {
        assert_eq!(extent.num_points(), values.len());

        Self::new(extent, Channel::new(values))
    }
}

impl<N, Chan> IndexedArray<N> for Array<N, Chan>
where
    N: ArrayIndexer<N>,
{
    type Indexer = N;

    #[inline]
    fn extent(&self) -> &ExtentN<N> {
        self.extent()
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    /// Set all points to the same value.
    #[inline]
    pub fn reset_values(&mut self, value: T)
    where
        T: Clone,
    {
        self.channels.fill(value);
    }
}

impl<'a, N, T, Store> IntoRawBytes<'a> for ArrayNx1<N, T, Store>
where
    T: 'static + Copy,
    Store: Deref<Target = [T]>,
{
    type Output = &'a [u8];

    fn into_raw_bytes(&'a self) -> Self::Output {
        self.channels.store().into_raw_bytes()
    }
}

impl<N, T> ArrayNx1<N, T, Vec<T>>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Creates a map that fills the entire `extent` with the same `value`.
    pub fn fill(extent: ExtentN<N>, value: T) -> Self
    where
        T: Clone,
    {
        Self::new(extent, Channel::new_fill(value, extent.num_points()))
    }

    /// Create a new array for `extent` where each point's value is determined by the `filler` function.
    pub fn fill_with(extent: ExtentN<N>, mut filler: impl FnMut(PointN<N>) -> T) -> Self
    where
        ArrayNx1<N, MaybeUninit<T>>:
            for<'r> ForEachMut<'r, N, PointN<N>, Item = &'r mut MaybeUninit<T>>,
    {
        let mut array = unsafe { ArrayNx1::maybe_uninit(extent) };

        array.for_each_mut(&extent, |p, x| unsafe {
            x.as_mut_ptr().write(filler(p));
        });

        unsafe { array.assume_init() }
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: Deref<Target = [T]>,
{
    /// Sets the extent minimum to `p`.
    #[inline]
    pub fn set_minimum(&mut self, p: PointN<N>) {
        self.extent.minimum = p;
    }

    /// Adds `p` to the extent minimum.
    #[inline]
    pub fn translate(&mut self, p: PointN<N>) {
        self.extent = self.extent.add(p);
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    PointN<N>: Point,
{
    /// Returns `true` iff this map contains point `p`.
    #[inline]
    pub fn contains(&self, p: PointN<N>) -> bool {
        self.extent.contains(p)
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    for<'r> Self: ForEachMut<'r, N, (), Item = &'r mut T>,
    PointN<N>: IntegerPoint<N>,
    Store: DerefMut<Target = [T]>,
{
    /// Fill the entire `extent` with the same `value`.
    pub fn fill_extent(&mut self, extent: &ExtentN<N>, value: T)
    where
        T: Clone,
    {
        if self.extent.eq(extent) {
            self.channels.fill(value);
        } else {
            self.for_each_mut(extent, |_: (), v| *v = value.clone());
        }
    }
}

impl<N, T> ArrayNx1<N, MaybeUninit<T>>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Creates an uninitialized map, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    pub unsafe fn maybe_uninit(extent: ExtentN<N>) -> Self {
        Self::new(extent, Channel::maybe_uninit(extent.num_points()))
    }

    /// Transmutes the map values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just reconstructs
    /// the internal `Vec` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the map must be initialized.
    pub unsafe fn assume_init(self) -> ArrayNx1<N, T> {
        let (extent, channel) = self.into_parts();
        let channel = channel.assume_init();

        ArrayNx1::new(extent, channel)
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, Chan> Get<Stride> for Array<N, Chan>
where
    Chan: Get<usize>,
{
    type Item = Chan::Item;

    #[inline]
    fn get(&self, stride: Stride) -> Self::Item {
        self.channels.get(stride.0)
    }
}

impl<'a, N, Chan> GetRef<'a, Stride> for Array<N, Chan>
where
    Chan: GetRef<'a, usize>,
{
    type Item = Chan::Item;

    #[inline]
    fn get_ref(&'a self, stride: Stride) -> Self::Item {
        self.channels.get_ref(stride.0)
    }
}

impl<'a, N, Chan> GetMut<'a, Stride> for Array<N, Chan>
where
    Chan: GetMut<'a, usize>,
{
    type Item = Chan::Item;

    #[inline]
    fn get_mut(&'a mut self, stride: Stride) -> Self::Item {
        self.channels.get_mut(stride.0)
    }
}

impl<N, Chan> Get<Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + Get<Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as Get<Stride>>::Item;

    #[inline]
    fn get(&self, p: Local<N>) -> Self::Item {
        self.get(self.stride_from_local_point(p))
    }
}

impl<'a, N, Chan> GetRef<'a, Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetRef<'a, Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as GetRef<'a, Stride>>::Item;

    #[inline]
    fn get_ref(&'a self, p: Local<N>) -> Self::Item {
        self.get_ref(self.stride_from_local_point(p))
    }
}

impl<'a, N, Chan> GetMut<'a, Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetMut<'a, Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as GetMut<'a, Stride>>::Item;

    #[inline]
    fn get_mut(&'a mut self, p: Local<N>) -> Self::Item {
        self.get_mut(self.stride_from_local_point(p))
    }
}

impl<N, Chan> Get<PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + Get<Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as Get<Local<N>>>::Item;

    #[inline]
    fn get(&self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get(Local(local_p))
    }
}

impl<'a, N, Chan> GetRef<'a, PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetRef<'a, Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as GetRef<'a, Local<N>>>::Item;

    #[inline]
    fn get_ref(&'a self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get_ref(Local(local_p))
    }
}

impl<'a, N, Chan> GetMut<'a, PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetMut<'a, Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as GetMut<'a, Local<N>>>::Item;

    #[inline]
    fn get_mut(&'a mut self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get_mut(Local(local_p))
    }
}

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

macro_rules! impl_array_for_each {
    (coords: $coords:ty; forwarder = |$p:ident, $stride:ident| $forward_coords:expr;) => {
        impl<N, T, Store> ForEach<N, $coords> for ArrayNx1<N, T, Store>
        where
            Self: Get<Stride, Item = T>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
        {
            type Item = T;

            #[inline]
            fn for_each(&self, iter_extent: &ExtentN<N>, mut f: impl FnMut($coords, T)) {
                let visitor = ArrayForEach::new_global(self.extent(), *iter_extent);
                visitor
                    .for_each_point_and_stride(|$p, $stride| f($forward_coords, self.get($stride)));
            }
        }

        impl<'a, N, T, Store> ForEachMut<'a, N, $coords> for ArrayNx1<N, T, Store>
        where
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
            T: 'a,
            Store: DerefMut<Target = [T]>,
        {
            type Item = &'a mut T;

            #[inline]
            fn for_each_mut(
                &'a mut self,
                iter_extent: &ExtentN<N>,
                mut f: impl FnMut($coords, &'a mut T),
            ) {
                let visitor = ArrayForEach::new_global(self.extent(), *iter_extent);
                visitor.for_each_point_and_stride(|$p, $stride| {
                    // Need to tell the borrow checker that we're handing out non-overlapping borrows.
                    f($forward_coords, unsafe {
                        &mut *self.channels.store_mut().as_mut_ptr().add($stride.0)
                    })
                });
            }
        }

        impl<'a, N, T, S, Store1, Store2> ForEachMut<'a, N, $coords>
            for (&mut ArrayNx1<N, T, Store1>, &mut ArrayNx1<N, S, Store2>)
        where
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
            T: 'a,
            S: 'a,
            Store1: DerefMut<Target = [T]>,
            Store2: DerefMut<Target = [S]>,
        {
            type Item = (&'a mut T, &'a mut S);

            // XXX: this is slightly weird because we don't actually need `&mut self` for this, but it works, so... ¯\_(ツ)_/¯
            #[inline]
            fn for_each_mut(
                &'a mut self,
                iter_extent: &ExtentN<N>,
                mut f: impl FnMut($coords, (&'a mut T, &'a mut S)),
            ) {
                let (s1, s2) = self;

                let visitor = ArrayForEach::new_global(s1.extent(), *iter_extent);
                visitor.for_each_point_and_stride(|$p, $stride| {
                    // Need to tell the borrow checker that we're handing out non-overlapping borrows.
                    f($forward_coords, unsafe {
                        (
                            &mut *s1.channels.store_mut().as_mut_ptr().add($stride.0),
                            &mut *s2.channels.store_mut().as_mut_ptr().add($stride.0),
                        )
                    })
                });
            }
        }
    };
}

impl_array_for_each!(
    coords: (PointN<N>, Stride);
    forwarder = |p, stride| (p, stride);
);
impl_array_for_each!(
    coords: Stride;
    forwarder = |_p, stride| stride;
);
impl_array_for_each!(
    coords: PointN<N>;
    forwarder = |p, _stride| p;
);
impl_array_for_each!(
    coords: ();
    forwarder = |_p, _stride| ();
);

//  ██████╗ ██████╗ ██████╗ ██╗   ██╗
// ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
// ██║     ██║   ██║██████╔╝ ╚████╔╝
// ██║     ██║   ██║██╔═══╝   ╚██╔╝
// ╚██████╗╚██████╔╝██║        ██║
//  ╚═════╝ ╚═════╝ ╚═╝        ╚═╝

// Newtype avoids potential conflicting impls downstream.
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ArrayCopySrc<Map>(pub Map);

impl<'a, N: 'a, T: 'a, Store: 'a> ReadExtent<'a, N> for ArrayNx1<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
{
    type Src = ArrayCopySrc<&'a ArrayNx1<N, T, Store>>;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let in_bounds_extent = extent.intersection(self.extent());

        once((in_bounds_extent, ArrayCopySrc(&self)))
    }
}

impl<'a, N, T, Store> WriteExtent<N, ArrayCopySrc<&'a Self>> for ArrayNx1<N, T, Store>
where
    Self: Get<Stride, Item = T> + for<'r> GetMut<'r, Stride, Item = &'r mut T>,
    N: ArrayIndexer<N>,
    T: Clone,
    PointN<N>: IntegerPoint<N>,
    ExtentN<N>: Copy,
    Store: Clone,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src_array: ArrayCopySrc<&'a Self>) {
        // It is assumed by the interface that extent is a subset of the src array, so we only need to intersect with the
        // destination.
        let in_bounds_extent = extent.intersection(self.extent());

        let copy_entire_array = in_bounds_extent.shape == self.extent().shape
            && in_bounds_extent.shape == src_array.0.extent().shape;

        if copy_entire_array {
            // Fast path, mostly for copying entire chunks between chunk maps.
            self.channels = src_array.0.channels.clone();
        } else {
            unchecked_copy_extent_between_arrays(self, src_array.0, &in_bounds_extent);
        }
    }
}

impl<'a, N, Out, Store, Map, F> WriteExtent<N, ArrayCopySrc<TransformMap<'a, Map, F>>>
    for ArrayNx1<N, Out, Store>
where
    Self: IndexedArray<N> + for<'r> GetMut<'r, Stride, Item = &'r mut Out>,
    Out: 'a,
    TransformMap<'a, Map, F>: IndexedArray<N> + Get<Stride, Item = Out>,
    PointN<N>: IntegerPoint<N>,
    ExtentN<N>: Copy,
{
    fn write_extent(
        &mut self,
        extent: &ExtentN<N>,
        src_array: ArrayCopySrc<TransformMap<'a, Map, F>>,
    ) {
        // It is assumed by the interface that extent is a subset of the src array, so we only need to intersect with the
        // destination.
        let in_bounds_extent = extent.intersection(self.extent());

        unchecked_copy_extent_between_arrays(self, &src_array.0, &in_bounds_extent);
    }
}

// SAFETY: `extent` must be in-bounds of both arrays.
fn unchecked_copy_extent_between_arrays<Dst, Src, N, T>(
    dst: &mut Dst,
    src: &Src,
    extent: &ExtentN<N>,
) where
    Dst: IndexedArray<N> + for<'r> GetMut<'r, Stride, Item = &'r mut T>,
    Src: IndexedArray<N> + Get<Stride, Item = T>,
    ExtentN<N>: Copy,
{
    let dst_extent = *dst.extent();
    // It shoudn't matter which type we use for the indexer.
    Dst::Indexer::for_each_stride_parallel_global_unchecked(
        &extent,
        &dst_extent,
        src.extent(),
        |s_dst, s_src| {
            // The actual copy.
            // PERF: could be faster with SIMD copy
            *dst.get_mut(s_dst) = src.get(s_src);
        },
    );
}

impl<N, T, Ch, Store> WriteExtent<N, ChunkCopySrc<N, T, Ch>> for ArrayNx1<N, T, Store>
where
    for<'r> Self: ForEachMut<'r, N, Stride, Item = &'r mut T>,
    Self: WriteExtent<N, ArrayCopySrc<Ch>>,
    N: ArrayIndexer<N>,
    T: 'static + Clone,
    PointN<N>: IntegerPoint<N>,
    Store: DerefMut<Target = [T]>,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: ChunkCopySrc<N, T, Ch>) {
        match src {
            Either::Left(array) => self.write_extent(extent, array),
            Either::Right(ambient) => self.fill_extent(extent, ambient.get()),
        }
    }
}

impl<N, T, Store, F> WriteExtent<N, F> for ArrayNx1<N, T, Store>
where
    for<'r> Self: ForEachMut<'r, N, PointN<N>, Item = &'r mut T>,
    F: Fn(PointN<N>) -> T,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: F) {
        self.for_each_mut(extent, |p, v| *v = (src)(p));
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
    use crate::{copy_extent, Array2x1, Array3x1, Get};

    use building_blocks_core::{Extent2, Extent3};

    #[test]
    fn fill_and_get_2d() {
        let extent = Extent2::from_min_and_shape(PointN([1, 1]), PointN([10, 10]));
        let mut array = Array2x1::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 100);
        *array.get_mut(Stride(0)) = 1;

        assert_eq!(array.get(Stride(0)), 1);
        assert_eq!(array.get_mut(Stride(0)), &mut 1);

        assert_eq!(array.get(Local(PointN([0, 0]))), 1);
        assert_eq!(array.get_mut(Local(PointN([0, 0]))), &mut 1);

        assert_eq!(array.get(PointN([1, 1])), 1);
        assert_eq!(array.get_mut(PointN([1, 1])), &mut 1);
    }

    #[test]
    fn fill_and_get_3d() {
        let extent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
        let mut array = Array3x1::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 1000);
        *array.get_mut(Stride(0)) = 1;

        assert_eq!(array.get(Stride(0)), 1);
        assert_eq!(array.get_mut(Stride(0)), &mut 1);

        assert_eq!(array.get(Local(Point3i::ZERO)), 1);
        assert_eq!(array.get_mut(Local(Point3i::ZERO)), &mut 1);

        assert_eq!(array.get(PointN([1, 1, 1])), 1);
        assert_eq!(array.get_mut(PointN([1, 1, 1])), &mut 1);
    }

    #[test]
    fn fill_and_for_each_2d() {
        let extent = Extent2::from_min_and_shape(Point2i::fill(1), Point2i::fill(10));
        let mut array = Array2x1::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 100);
        *array.get_mut(Stride(0)) = 1;

        array.for_each(&extent, |p: Point2i, value| {
            if p == Point2i::fill(1) {
                assert_eq!(value, 1);
            } else {
                assert_eq!(value, 0);
            }
        });
    }

    #[test]
    fn fill_and_for_each_3d() {
        let extent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
        let mut array = Array3x1::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 1000);
        *array.get_mut(Stride(0)) = 1;

        array.for_each(&extent, |p: Point3i, value| {
            if p == Point3i::fill(1) {
                assert_eq!(value, 1);
            } else {
                assert_eq!(value, 0);
            }
        });
    }

    #[test]
    fn uninitialized() {
        let extent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
        let mut array: Array3x1<MaybeUninit<i32>> = unsafe { Array3x1::maybe_uninit(extent) };

        for p in extent.iter_points() {
            unsafe {
                array.get_mut(p).as_mut_ptr().write(1);
            }
        }

        let array = unsafe { array.assume_init() };

        for p in extent.iter_points() {
            assert_eq!(array.get(p), 1i32);
        }
    }

    #[test]
    fn copy() {
        let extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x1::fill(extent, 0);

        let subextent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(5));
        for p in subextent.iter_points() {
            *array.get_mut(p) = p.x() + p.y() + p.z();
        }

        let mut other_array = Array3x1::fill(extent, 0);
        copy_extent(&subextent, &array, &mut other_array);

        assert_eq!(array, other_array);
    }

    #[test]
    fn zipped_mut_iter() {
        let extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array1 = Array3x1::fill(extent, 0);
        let mut array2 = Array3x1::fill(extent, false);

        (&mut array1, &mut array2).for_each_mut(&extent, |_p: Point3i, (val1, val2)| {
            *val1 = 1;
            *val2 = true;
        });
    }

    #[test]
    fn multichannel_get() {
        let extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let ch1 = Channel::new_fill(0, extent.num_points());
        let ch2 = Channel::new_fill(1, extent.num_points());
        let mut array = Array::new(extent, (ch1, ch2));

        assert_eq!(array.get(Stride(0)), (0, 1));
        assert_eq!(array.get_ref(Stride(0)), (&0, &1));
        assert_eq!(array.get_mut(Stride(0)), (&mut 0, &mut 1));

        assert_eq!(array.get(Local(Point3i::fill(0))), (0, 1));
        assert_eq!(array.get_ref(Local(Point3i::fill(0))), (&0, &1));
        assert_eq!(array.get_mut(Local(Point3i::fill(0))), (&mut 0, &mut 1));

        assert_eq!(array.get(Point3i::fill(0)), (0, 1));
        assert_eq!(array.get_ref(Point3i::fill(0)), (&0, &1));
        assert_eq!(array.get_mut(Point3i::fill(0)), (&mut 0, &mut 1));
    }
}
