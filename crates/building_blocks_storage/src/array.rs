//! N-dimensional arrays, where N is 2 or 3.
//!
//! The domains of all arrays are located within an ambient space, a signed integer lattice where the elements are `Point2i` or
//! `Point3i`. This means they contain data at exactly the set of points in an `ExtentN`, and no more.
//!
//! # Indexing
//!
//! You can index an array with 3 kinds of coordinates, with traits:
//!   - `Get*<Stride>`: flat array offset
//!   - `Get*<&LocalN>`: N-dimensional point in extent-local coordinates (i.e. min = `[0, 0, 0]`)
//!   - `Get*<PointN>`: N-dimensional point in global (ambient) coordinates
//!
//! Indexing assumes that the coordinates are in-bounds of the array, panicking otherwise.
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

mod array2;
mod array3;
mod compression;

#[cfg(feature = "dot_vox")]
mod dot_vox_conversions;
#[cfg(feature = "image")]
mod image_conversions;

pub use array2::Array2;
pub use array3::Array3;
pub use compression::{FastArrayCompression, FastCompressedArray};

use crate::{
    chunk_map::ChunkCopySrc, ForEach, ForEachMut, ForEachRef, Get, GetMut, GetRef, GetUnchecked,
    GetUncheckedMut, GetUncheckedMutRelease, GetUncheckedRef, GetUncheckedRefRelease,
    GetUncheckedRelease, ReadExtent, TransformMap, WriteExtent,
};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Deref, DerefMut, Mul, Sub, SubAssign};
use either::Either;
use num::Zero;
use serde::{Deserialize, Serialize};

/// When a lattice map implements `Array`, that means there is some underlying array with the
/// location and shape dictated by the extent.
///
/// For the sake of generic impls, if the same map also implements `Get*<Stride>`, it must use the
/// same data layout as `ArrayN`.
pub trait Array<N> {
    type Indexer: ArrayIndexer<N>;

    fn extent(&self) -> &ExtentN<N>;

    #[inline]
    fn for_each_point_and_stride(&self, extent: &ExtentN<N>, f: impl FnMut(PointN<N>, Stride))
    where
        PointN<N>: IntegerPoint<N>,
    {
        Self::Indexer::for_each_point_and_stride_global(self.extent(), extent, f);
    }

    #[inline]
    unsafe fn for_each_point_and_stride_global_unchecked(
        &self,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        Self::Indexer::for_each_point_and_stride_global_unchecked(self.extent(), iter_extent, f);
    }

    #[inline]
    unsafe fn for_each_point_and_stride_local_unchecked(
        &self,
        iter_min: Local<N>,
        iter_shape: PointN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: Copy,
    {
        Self::Indexer::for_each_point_and_stride_local_unchecked(
            self.extent().shape,
            iter_min,
            iter_shape,
            f,
        );
    }

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

pub trait ArrayIndexer<N> {
    fn stride_from_local_point(shape: PointN<N>, point: Local<N>) -> Stride;

    fn for_each_point_and_stride_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_min: PointN<N>,
        iter_shape: PointN<N>,
        f: impl FnMut(PointN<N>, Stride),
    );

    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &ExtentN<N>,
        array1_extent: &ExtentN<N>,
        array2_extent: &ExtentN<N>,
        f: impl FnMut(Stride, Stride),
    );

    #[inline]
    fn for_each_point_and_stride_local_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_shape: PointN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: Copy,
    {
        Self::for_each_point_and_stride_unchecked(
            array_shape,
            index_min,
            index_min.0,
            iter_shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_local(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(&ExtentN::from_min_and_shape(
            PointN::ZERO,
            array_extent.shape,
        ));

        Self::for_each_point_and_stride_local_unchecked(
            array_extent.shape,
            Local(iter_extent.minimum),
            iter_extent.shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_global_unchecked(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Translate to local coordinates.
        let index_min = iter_extent.minimum - array_extent.minimum;

        Self::for_each_point_and_stride_unchecked(
            array_extent.shape,
            Local(index_min),
            iter_extent.minimum,
            iter_extent.shape,
            f,
        );
    }

    #[inline]
    fn for_each_point_and_stride_global(
        array_extent: &ExtentN<N>,
        iter_extent: &ExtentN<N>,
        f: impl FnMut(PointN<N>, Stride),
    ) where
        PointN<N>: IntegerPoint<N>,
    {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(array_extent);

        Self::for_each_point_and_stride_global_unchecked(array_extent, &iter_extent, f);
    }

    #[inline]
    fn strides_from_local_points(shape: PointN<N>, points: &[Local<N>], strides: &mut [Stride])
    where
        PointN<N>: Copy,
    {
        for (i, p) in points.iter().enumerate() {
            strides[i] = Self::stride_from_local_point(shape, *p);
        }
    }
}

/// A map from lattice location `PointN<N>` to data `T`, stored as a flat array on the heap.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArrayN<N, T, Store = Vec<T>> {
    values: Store,
    extent: ExtentN<N>,
    marker: std::marker::PhantomData<T>,
}

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
        for v in self.values.as_mut().iter_mut() {
            *v = value.clone();
        }
    }
}

impl<N, T, Store> ArrayN<N, T, Store>
where
    Store: Deref<Target = [T]>,
{
    /// Returns the slice of values, reinterpreted as raw bytes.
    #[inline]
    pub fn bytes_slice(&self) -> &[u8]
    where
        T: Copy,
    {
        unsafe {
            std::slice::from_raw_parts(
                self.values.as_ptr() as *const u8,
                self.values.len() * core::mem::size_of::<T>(),
            )
        }
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
    /// Create a new `ArrayN` directly from the extent and values. This asserts that the
    /// number of points in the extent matches the length of the values `Vec`.
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
    Self: ForEachMut<N, Stride, Data = T>,
    PointN<N>: IntegerPoint<N>,
    ExtentN<N>: PartialEq,
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
            self.for_each_mut(extent, |_s: Stride, v| *v = value.clone());
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

/// Map-local coordinates.
///
/// Most commonly, you will index a lattice map with a `PointN<N>`, which is assumed to be in global
/// coordinates. `Local<N>` only applies to lattice maps where a point must first be translated from
/// global coordinates into map-local coordinates before indexing with `Get<Local<N>>`.
#[derive(Debug, Eq, PartialEq)]
pub struct Local<N>(pub PointN<N>);

impl<N> Clone for Local<N>
where
    PointN<N>: Clone,
{
    fn clone(&self) -> Self {
        Local(self.0.clone())
    }
}
impl<N> Copy for Local<N> where PointN<N>: Copy {}

impl<N> Local<N> {
    /// Wraps all of the `points` using the `Local` constructor.
    #[inline]
    pub fn localize_points(points: &[PointN<N>]) -> Vec<Local<N>>
    where
        PointN<N>: Clone,
    {
        points.iter().cloned().map(Local).collect()
    }
}

impl<N> Deref for Local<N> {
    type Target = PointN<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The most efficient coordinates for slice-backed lattice maps. A single number that translates
/// directly to a slice offset.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Stride(pub usize);

impl Zero for Stride {
    #[inline]
    fn zero() -> Self {
        Stride(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Add for Stride {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        // Wraps for negative point offsets.
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Stride {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        // Wraps for negative point offsets.
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul<usize> for Stride {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: usize) -> Self::Output {
        Self(self.0.wrapping_mul(rhs))
    }
}

impl AddAssign for Stride {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Stride {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

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
            Self: Sized + Get<Stride, Data = T> + GetUnchecked<Stride, Data = T>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
        {
            type Data = T;

            #[inline]
            fn for_each(&self, iter_extent: &ExtentN<N>, mut f: impl FnMut($coords, T)) {
                self.for_each_point_and_stride(iter_extent, |$p, $stride| {
                    f($forward_coords, self.get_unchecked_release($stride))
                })
            }
        }

        impl<N, T, Store> ForEachRef<N, $coords> for ArrayN<N, T, Store>
        where
            Self: Sized + GetRef<Stride, Data = T> + GetUncheckedRef<Stride, Data = T>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
        {
            type Data = T;

            #[inline]
            fn for_each_ref(&self, iter_extent: &ExtentN<N>, mut f: impl FnMut($coords, &T)) {
                self.for_each_point_and_stride(iter_extent, |$p, $stride| {
                    f($forward_coords, self.get_unchecked_ref_release($stride))
                })
            }
        }

        impl<N, T, Store> ForEachMut<N, $coords> for ArrayN<N, T, Store>
        where
            Self: Sized + GetMut<Stride, Data = T> + GetUncheckedMut<Stride, Data = T>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint<N>,
            ExtentN<N>: Copy,
        {
            type Data = T;

            #[inline]
            fn for_each_mut(
                &mut self,
                iter_extent: &ExtentN<N>,
                mut f: impl FnMut($coords, &mut T),
            ) {
                // Can't borrow self as mutable and immutable at the same time.
                let array_extent = *self.extent();
                <Self as Array<N>>::Indexer::for_each_point_and_stride_global(
                    &array_extent,
                    iter_extent,
                    |$p, $stride| f($forward_coords, self.get_unchecked_mut_release($stride)),
                )
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
        // It is assumed by the interface that extent is a subset of the src array, so we only need
        // to intersect with the destination.
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
        // It is assumed by the interface that extent is a subset of the src array, so we only need
        // to intersect with the destination.
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

impl<Map, N, T, Store> WriteExtent<N, ChunkCopySrc<Map, N, T>> for ArrayN<N, T, Store>
where
    Self: ForEachMut<N, Stride, Data = T> + WriteExtent<N, ArrayCopySrc<Map>>,
    N: ArrayIndexer<N>,
    T: Clone,
    PointN<N>: IntegerPoint<N>,
    ExtentN<N>: PartialEq,
    Store: DerefMut<Target = [T]>,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: ChunkCopySrc<Map, N, T>) {
        match src {
            Either::Left(array) => self.write_extent(extent, array),
            Either::Right(ambient) => self.fill_extent(extent, ambient.get()),
        }
    }
}

impl<'a, N, T: 'a + Clone, Store, F> WriteExtent<N, F> for ArrayN<N, T, Store>
where
    Self: ForEachMut<N, PointN<N>, Data = T>,
    F: Fn(PointN<N>) -> T,
    PointN<N>: IntegerPoint<N>,
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
    use crate::{access::GetUnchecked, copy_extent, Array2, Array3, Get};

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
}
