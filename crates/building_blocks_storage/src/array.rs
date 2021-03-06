//! N-dimensional arrays, where N is 2 or 3.
//!
//! The domains of all arrays are located within an ambient space, a signed integer lattice where the elements are `Point2i` or
//! `Point3i`. This means they contain data at exactly the set of points in an `ExtentN`, and no more.
//!
//! # Indexing
//!
//! You can index an array with 3 kinds of coordinates, with [`Get`](crate::access) traits:
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
//! let mut array = Array3::fill(array_extent, 0);
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
//! let mut array = Array3::fill_with(extent, |p| if p.x() % 2 == 0 { 1 } else { 0 });
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
//! By default, `ArrayN` uses a `Vec` to store elements. But any type that implements `Deref<Target = [T]>` or `DerefMut<Target
//! = [T]>` should be usable. This means you can construct an array with most pointer types.
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(64));
//! // Borrow `array`'s values for the lifetime of `other_array`.
//! let array = Array3::fill(extent, 1);
//! let other_array = Array3::new(extent, array.values_slice());
//! assert_eq!(other_array.get(Stride(0)), 1);
//!
//! // A stack-allocated array.
//! let mut data = [1; 64 * 64 * 64];
//! let mut stack_array = Array3::new(extent, &mut data[..]);
//! *stack_array.get_mut(Stride(0)) = 2;
//! assert_eq!(data[0], 2);
//!
//! // A boxed array.
//! let data: Box<[u32]> = Box::new([1; 64 * 64 * 64]); // must forget the size
//! let box_array = Array3::new(extent, data);
//! box_array.for_each(&extent, |p: Point3i, value| assert_eq!(value, 1));
//! ```

mod compression;
mod coords;
mod for_each;
#[macro_use]
mod for_each2;
#[macro_use]
mod for_each3;
mod indexer;

#[cfg(feature = "dot_vox")]
mod dot_vox_conversions;
#[cfg(feature = "image")]
mod image_conversions;

pub use compression::{FastArrayCompression, FastCompressedArray};
pub use coords::*;
pub use for_each::*;
pub use indexer::*;

pub(crate) use for_each2::{for_each_stride_parallel_global_unchecked2, Array2ForEachState};
pub(crate) use for_each3::{for_each_stride_parallel_global_unchecked3, Array3ForEachState};

use crate::{
    ChunkCopySrc, ForEach, ForEachMut, Get, GetMut, GetRef, GetUnchecked, GetUncheckedMut,
    GetUncheckedMutRelease, GetUncheckedRef, GetUncheckedRelease, IntoRawBytes, ReadExtent,
    TransformMap, WriteExtent,
};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};
use core::mem::MaybeUninit;
use core::ops::{Add, Deref, DerefMut};
use either::Either;
use serde::{Deserialize, Serialize};

/// When a lattice map implements `Array`, that means there is some underlying array with the location and shape dictated by the
/// extent.
///
/// For the sake of generic impls, if the same map also implements `Get*<Stride>`, it must use the same data layout as `ArrayN`.
pub trait Array<N> {
    type Indexer: ArrayIndexer<N>;

    fn extent(&self) -> &ExtentN<N>;

    #[inline]
    fn stride_from_local_point(&self, p: Local<N>) -> Stride
    where
        PointN<N>: Copy,
    {
        Self::Indexer::stride_from_local_point(self.extent().shape, p)
    }

    #[inline]
    fn strides_from_local_points(&self, points: &[Local<N>], strides: &mut [Stride])
    where
        PointN<N>: Copy,
    {
        Self::Indexer::strides_from_local_points(self.extent().shape, points, strides)
    }
}

/// A map from lattice location `PointN<N>` to data `T`, stored as a flat array on the heap.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArrayN<N, T, Store = Vec<T>> {
    values: Store,
    extent: ExtentN<N>,
    marker: std::marker::PhantomData<T>,
}

/// A 2-dimensional `Array`.
pub type Array2<T, Store = Vec<T>> = ArrayN<[i32; 2], T, Store>;
/// A 3-dimensional `Array`.
pub type Array3<T, Store = Vec<T>> = ArrayN<[i32; 3], T, Store>;

impl<N, T, Store> ArrayN<N, T, Store> {
    /// Moves the raw extent and values storage out of `self`.
    #[inline]
    pub fn into_parts(self) -> (ExtentN<N>, Store) {
        (self.extent, self.values)
    }

    #[inline]
    pub fn extent(&self) -> &ExtentN<N> {
        &self.extent
    }
}

impl<N, T, Store> Array<N> for ArrayN<N, T, Store>
where
    N: ArrayIndexer<N>,
{
    type Indexer = N;

    #[inline]
    fn extent(&self) -> &ExtentN<N> {
        self.extent()
    }
}

impl<N, T, Store> ArrayN<N, T, Store>
where
    Store: Deref<Target = [T]>,
{
    /// Returns the entire slice of values.
    #[inline]
    pub fn values_slice(&self) -> &[T] {
        self.values.as_ref()
    }
}

impl<N, T, Store> ArrayN<N, T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    /// Returns the entire slice of values.
    #[inline]
    pub fn values_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut()
    }

    /// Set all points to the same value.
    #[inline]
    pub fn reset_values(&mut self, value: T)
    where
        T: Clone,
    {
        self.values.fill(value);
    }
}

impl<'a, N, T, Store> IntoRawBytes<'a> for ArrayN<N, T, Store>
where
    T: 'static + Copy,
    Store: Deref<Target = [T]>,
{
    type Output = &'a [u8];

    fn into_raw_bytes(&'a self) -> Self::Output {
        self.values.into_raw_bytes()
    }
}

impl<N, T> ArrayN<N, T, Vec<T>>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Creates a map that fills the entire `extent` with the same `value`.
    pub fn fill(extent: ExtentN<N>, value: T) -> Self
    where
        T: Clone,
    {
        Self::new(extent, vec![value; extent.num_points()])
    }

    /// Create a new array for `extent` where each point's value is determined by the `filler` function.
    pub fn fill_with(extent: ExtentN<N>, mut filler: impl FnMut(PointN<N>) -> T) -> Self
    where
        ArrayN<N, MaybeUninit<T>>: GetMut<PointN<N>, Data = MaybeUninit<T>>,
    {
        let mut array = unsafe { ArrayN::maybe_uninit(extent) };

        for p in extent.iter_points() {
            unsafe {
                array.get_mut(p).as_mut_ptr().write(filler(p));
            }
        }

        unsafe { array.assume_init() }
    }
}

impl<N, T, Store> ArrayN<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
    Store: Deref<Target = [T]>,
{
    /// Create a new `ArrayN` directly from the extent and values. This asserts that the number of points in the extent matches
    /// the length of the values `Vec`.
    pub fn new(extent: ExtentN<N>, values: Store) -> Self {
        assert_eq!(extent.num_points(), values.len());

        Self {
            values,
            extent,
            marker: Default::default(),
        }
    }

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

impl<N, T, Store> ArrayN<N, T, Store>
where
    PointN<N>: Point,
{
    /// Returns `true` iff this map contains point `p`.
    #[inline]
    pub fn contains(&self, p: PointN<N>) -> bool {
        self.extent.contains(p)
    }
}

impl<N, T, Store> ArrayN<N, T, Store>
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
            self.values.fill(value);
        } else {
            self.for_each_mut(extent, |_: (), v| *v = value.clone());
        }
    }
}

impl<N, T> ArrayN<N, MaybeUninit<T>>
where
    PointN<N>: IntegerPoint<N>,
{
    /// Creates an uninitialized map, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    pub unsafe fn maybe_uninit(extent: ExtentN<N>) -> Self {
        let num_points = extent.num_points();
        let mut values = Vec::with_capacity(num_points);
        values.set_len(num_points);

        Self::new(extent, values)
    }

    /// Transmutes the map values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just reconstructs
    /// the internal `Vec` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the map must be initialized.
    pub unsafe fn assume_init(self) -> ArrayN<N, T> {
        let transmuted_values = {
            // Ensure the original vector is not dropped.
            let mut v_clone = core::mem::ManuallyDrop::new(self.values);

            Vec::from_raw_parts(
                v_clone.as_mut_ptr() as *mut T,
                v_clone.len(),
                v_clone.capacity(),
            )
        };

        ArrayN::new(self.extent, transmuted_values)
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<N, T, Store> GetRef<Stride> for ArrayN<N, T, Store>
where
    Store: Deref<Target = [T]>,
{
    type Data = T;

    #[inline]
    fn get_ref(&self, stride: Stride) -> &Self::Data {
        &self.values[stride.0]
    }
}

impl<N, T, Store> GetUncheckedRef<Stride> for ArrayN<N, T, Store>
where
    Store: Deref<Target = [T]>,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_ref(&self, stride: Stride) -> &Self::Data {
        self.values.get_unchecked(stride.0)
    }
}

impl<N, T, Store> GetMut<Stride> for ArrayN<N, T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, stride: Stride) -> &mut Self::Data {
        &mut self.values[stride.0]
    }
}

impl<N, T, Store> GetUncheckedMut<Stride> for ArrayN<N, T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, stride: Stride) -> &mut Self::Data {
        self.values.get_unchecked_mut(stride.0)
    }
}

impl<N, T, Store> GetRef<Local<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetRef<Stride, Data = T>,
    PointN<N>: Copy,
{
    type Data = T;

    #[inline]
    fn get_ref(&self, p: Local<N>) -> &Self::Data {
        self.get_ref(self.stride_from_local_point(p))
    }
}

impl<N, T, Store> GetMut<Local<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetMut<Stride, Data = T>,
    PointN<N>: Copy,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, p: Local<N>) -> &mut Self::Data {
        self.get_mut(self.stride_from_local_point(p))
    }
}

impl<N, T, Store> GetRef<PointN<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetRef<Local<N>, Data = T>,
    PointN<N>: Point,
{
    type Data = T;

    #[inline]
    fn get_ref(&self, p: PointN<N>) -> &Self::Data {
        let local_p = p - self.extent().minimum;

        self.get_ref(Local(local_p))
    }
}

impl<N, T, Store> GetMut<PointN<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetMut<Local<N>, Data = T>,
    PointN<N>: Point,
{
    type Data = T;

    #[inline]
    fn get_mut(&mut self, p: PointN<N>) -> &mut Self::Data {
        let local_p = p - self.extent().minimum;

        GetMut::<Local<N>>::get_mut(self, Local(local_p))
    }
}

impl<N, T, Store> GetUncheckedRef<Local<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetUncheckedRef<Stride, Data = T>,
    PointN<N>: Copy,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_ref(&self, p: Local<N>) -> &Self::Data {
        self.get_unchecked_ref(self.stride_from_local_point(p))
    }
}

impl<N, T, Store> GetUncheckedMut<Local<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetUncheckedMut<Stride, Data = T>,
    PointN<N>: Copy,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, p: Local<N>) -> &mut Self::Data {
        self.get_unchecked_mut(self.stride_from_local_point(p))
    }
}

impl<N, T, Store> GetUncheckedRef<PointN<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetUncheckedRef<Local<N>, Data = T>,
    PointN<N>: Point,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_ref(&self, p: PointN<N>) -> &Self::Data {
        let local_p = p - self.extent().minimum;

        self.get_unchecked_ref(Local(local_p))
    }
}

impl<N, T, Store> GetUncheckedMut<PointN<N>> for ArrayN<N, T, Store>
where
    Self: Array<N> + GetUncheckedMut<Local<N>, Data = T>,
    PointN<N>: Point,
{
    type Data = T;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, p: PointN<N>) -> &mut Self::Data {
        let local_p = p - self.extent().minimum;

        GetUncheckedMut::<Local<N>>::get_unchecked_mut(self, Local(local_p))
    }
}

impl_get_via_get_ref_and_clone!(ArrayN<N, T, Store>, N, T, Store);

// ███████╗ ██████╗ ██████╗     ███████╗ █████╗  ██████╗██╗  ██╗
// ██╔════╝██╔═══██╗██╔══██╗    ██╔════╝██╔══██╗██╔════╝██║  ██║
// █████╗  ██║   ██║██████╔╝    █████╗  ███████║██║     ███████║
// ██╔══╝  ██║   ██║██╔══██╗    ██╔══╝  ██╔══██║██║     ██╔══██║
// ██║     ╚██████╔╝██║  ██║    ███████╗██║  ██║╚██████╗██║  ██║
// ╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

macro_rules! impl_array_for_each {
    (coords: $coords:ty; forwarder = |$p:ident, $stride:ident| $forward_coords:expr;) => {
        impl<N, T, Store> ForEach<N, $coords> for ArrayN<N, T, Store>
        where
            Self: Get<Stride, Data = T> + GetUnchecked<Stride, Data = T>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
        {
            type Item = T;

            #[inline]
            fn for_each(&self, iter_extent: &ExtentN<N>, mut f: impl FnMut($coords, T)) {
                let visitor = ArrayForEach::new_global(self.extent(), *iter_extent);
                visitor.for_each_point_and_stride(|$p, $stride| {
                    f($forward_coords, self.get_unchecked_release($stride))
                });
            }
        }

        impl<'a, N, T, Store> ForEachMut<'a, N, $coords> for ArrayN<N, T, Store>
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
                        &mut *self.values.as_mut_ptr().add($stride.0)
                    })
                });
            }
        }

        impl<'a, N, T, S, Store1, Store2> ForEachMut<'a, N, $coords>
            for (&mut ArrayN<N, T, Store1>, &mut ArrayN<N, S, Store2>)
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
                            &mut *s1.values.as_mut_ptr().add($stride.0),
                            &mut *s2.values.as_mut_ptr().add($stride.0),
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

impl<'a, N: 'a, T: 'a, Store: 'a> ReadExtent<'a, N> for ArrayN<N, T, Store>
where
    PointN<N>: IntegerPoint<N>,
{
    type Src = ArrayCopySrc<&'a ArrayN<N, T, Store>>;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let in_bounds_extent = extent.intersection(self.extent());

        once((in_bounds_extent, ArrayCopySrc(&self)))
    }
}

impl<'a, N, T, Store> WriteExtent<N, ArrayCopySrc<&'a Self>> for ArrayN<N, T, Store>
where
    Self: GetUncheckedRelease<Stride, T> + GetUncheckedMutRelease<Stride, T>,
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
            self.values = src_array.0.values.clone();
        } else {
            unchecked_copy_extent_between_arrays(self, src_array.0, &in_bounds_extent);
        }
    }
}

impl<'a, N, T, Store, Map, F> WriteExtent<N, ArrayCopySrc<TransformMap<'a, Map, F>>>
    for ArrayN<N, T, Store>
where
    Self: Array<N> + GetUncheckedMutRelease<Stride, T>,
    T: Clone,
    TransformMap<'a, Map, F>: Array<N> + GetUncheckedRelease<Stride, T>,
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
    Dst: Array<N> + GetUncheckedMutRelease<Stride, T>,
    Src: Array<N> + GetUncheckedRelease<Stride, T>,
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
            *dst.get_unchecked_mut_release(s_dst) = src.get_unchecked_release(s_src);
        },
    );
}

impl<N, T, Ch, Store> WriteExtent<N, ChunkCopySrc<N, T, Ch>> for ArrayN<N, T, Store>
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

impl<N, T, Store, F> WriteExtent<N, F> for ArrayN<N, T, Store>
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
    use crate::{copy_extent, Array2, Array3, Get, GetUnchecked};

    use building_blocks_core::{Extent2, Extent3};

    #[test]
    fn fill_and_get_2d() {
        let extent = Extent2::from_min_and_shape(PointN([1, 1]), PointN([10, 10]));
        let mut array = Array2::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 100);
        *array.get_mut(Stride(0)) = 1;

        assert_eq!(array.get(Stride(0)), 1);
        assert_eq!(array.get_mut(Stride(0)), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(Stride(0)) }, 1);
        assert_eq!(unsafe { array.get_unchecked_mut(Stride(0)) }, &mut 1);

        assert_eq!(array.get(Local(PointN([0, 0]))), 1);
        assert_eq!(array.get_mut(Local(PointN([0, 0]))), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(Local(PointN([0, 0]))) }, 1);
        assert_eq!(
            unsafe { array.get_unchecked_mut(Local(PointN([0, 0]))) },
            &mut 1
        );

        assert_eq!(array.get(PointN([1, 1])), 1);
        assert_eq!(array.get_mut(PointN([1, 1])), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(PointN([1, 1])) }, 1);
        assert_eq!(unsafe { array.get_unchecked_mut(PointN([1, 1])) }, &mut 1);
    }

    #[test]
    fn fill_and_get_3d() {
        let extent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
        let mut array = Array3::fill(extent, 0);
        assert_eq!(array.extent.num_points(), 1000);
        *array.get_mut(Stride(0)) = 1;

        assert_eq!(array.get(Stride(0)), 1);
        assert_eq!(array.get_mut(Stride(0)), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(Stride(0)) }, 1);
        assert_eq!(unsafe { array.get_unchecked_mut(Stride(0)) }, &mut 1);

        assert_eq!(array.get(Local(Point3i::ZERO)), 1);
        assert_eq!(array.get_mut(Local(Point3i::ZERO)), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(Local(Point3i::ZERO)) }, 1);
        assert_eq!(
            unsafe { array.get_unchecked_mut(Local(Point3i::ZERO)) },
            &mut 1
        );

        assert_eq!(array.get(PointN([1, 1, 1])), 1);
        assert_eq!(array.get_mut(PointN([1, 1, 1])), &mut 1);
        assert_eq!(unsafe { array.get_unchecked(PointN([1, 1, 1])) }, 1);
        assert_eq!(
            unsafe { array.get_unchecked_mut(PointN([1, 1, 1])) },
            &mut 1
        );
    }

    #[test]
    fn fill_and_for_each_2d() {
        let extent = Extent2::from_min_and_shape(Point2i::fill(1), Point2i::fill(10));
        let mut array = Array2::fill(extent, 0);
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
        let mut array = Array3::fill(extent, 0);
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
        let mut array: Array3<MaybeUninit<i32>> = unsafe { Array3::maybe_uninit(extent) };

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
        let mut array = Array3::fill(extent, 0);

        let subextent = Extent3::from_min_and_shape(Point3i::fill(1), Point3i::fill(5));
        for p in subextent.iter_points() {
            *array.get_mut(p) = p.x() + p.y() + p.z();
        }

        let mut other_array = Array3::fill(extent, 0);
        copy_extent(&subextent, &array, &mut other_array);

        assert_eq!(array, other_array);
    }

    #[test]
    fn multichannel_mut_iter() {
        let extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array1 = Array3::fill(extent, 0);
        let mut array2 = Array3::fill(extent, false);

        (&mut array1, &mut array2).for_each_mut(&extent, |_p: Point3i, (val1, val2)| {
            *val1 = 1;
            *val2 = true;
        });
    }
}
