//! N-dimensional arrays, where N is 2 or 3.
//!
//! The domains of all arrays are located within an ambient space, a signed integer lattice where the elements are `Point2i` or
//! `Point3i`. This means they contain data at exactly the set of points in an `ExtentN`, and no more.
//!
//! # Indexing
//!
//! You can index an array with 3 kinds of coordinates, with [`Get`](crate::access_traits) traits:
//!   - `Get*<Stride>`: flat array offset
//!   - `Get*<Local<N>>`: N-dimensional point in extent-local coordinates (i.e. min = `[0, 0, 0]`)
//!   - `Get*<PointN<N>>`: N-dimensional point in global (ambient) coordinates
//!
//! The safe traits will panic on out-of-bounds access, but this has a significant runtime cost. For faster unchecked access,
//! you can use the unsafe `Get*Unchecked` traits.
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
//! let offsets = Local::localize_points_array(&Point3i::VON_NEUMANN_OFFSETS);
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
//! # Single-Channel Array Storage
//!
//! A single-channel `Array` is backed by a `Channel<T, Store>` where `Store` is some kind of fat pointer. By default,
//! `ArrayNx1<T>` uses a boxed slice for storage.
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::prelude::*;
//! # use std::sync::Arc;
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(32));
//! // Borrow `array`'s values for the lifetime of `other_array`.
//! let array = Array3x1::fill(extent, 1);
//! let other_array = Array3x1::new_one_channel(extent, array.channels().store().as_ref());
//! assert_eq!(other_array.get(Stride(0)), 1);
//!
//! // A stack-allocated array.
//! let mut data = [1; 32 * 32 * 32];
//! let mut stack_array = Array3x1::new_one_channel(extent, &mut data[..]);
//! *stack_array.get_mut(Stride(0)) = 2;
//! assert_eq!(data[0], 2);
//!
//! // An Arc array.
//! let data: Arc<[u32]> = Arc::new([1; 32 * 32 * 32]); // must forget the size
//! let box_array = Array3x1::new_one_channel(extent, data);
//! box_array.for_each(&extent, |p: Point3i, value| assert_eq!(value, 1));
//! ```
//!
//! # Multichannel
//!
//! It's often the case that you have multiple data types to store per spatial dimension. For example, you might store geometry
//! data like `Sd8` as well as a voxel type identifier. While you can put these in a struct, that may not be the most efficient
//! option. If you only need access to one of those fields of the struct for a particular algorithm, then you will needlessly
//! load the entire struct into cache. To avoid this problem, `Array` supports storing multiple data "channels" in
//! structure-of-arrays (`SoA`) style.
//!
//! ```
//! # use building_blocks_core::prelude::*;
//! # use building_blocks_storage::{prelude::*};
//! # let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
//! #
//! #[derive(Clone, Copy, Debug, Eq, PartialEq)]
//! struct VoxelId(u8);
//!
//! // This means 3D with 2 channels. Although we pass in a tuple, the two data
//! // types are internally stored in separate arrays.
//! let mut array = Array3x2::fill(extent, (VoxelId(0), 1.0));
//!
//! // This array supports all of the usual access traits and maps the channels
//! // to tuples as you would expect.
//! let p = Point3i::fill(1);
//! assert_eq!(array.get(p), (VoxelId(0), 1.0));
//! assert_eq!(array.get_ref(p), (&VoxelId(0), &1.0));
//! assert_eq!(array.get_mut(p), (&mut VoxelId(0), &mut 1.0));
//!
//! // Here we choose to access just one channel, and there is no performance penalty.
//! array.borrow_channels_mut(|(_id, dist)| dist).for_each_mut(&extent, |p: Point3i, dist| {
//!     let r = p.dot(p);
//!     *dist = (r as f32).sqrt();
//! });
//!
//! // And if we want to copy just one of those channels into another map, we can
//! // create a new array that borrows just one of the channels.
//! let mut dst = Array3x1::fill(extent, 0.0);
//! let src = array.borrow_channels(|(_id, dist)| dist);
//! copy_extent(&extent, &src, &mut dst);
//! ```

mod coords;
mod indexer;

pub mod channels;
pub mod compression;
#[macro_use]
pub mod for_each;

#[cfg(feature = "dot_vox")]
mod dot_vox_conversions;
#[cfg(feature = "image")]
mod image_conversions;

pub use channels::Channel;
pub use coords::*;
pub use indexer::IndexedArray;

pub(crate) use channels::*;
pub(crate) use for_each::*;
pub(crate) use indexer::*;

use crate::{
    chunk_tree::ChunkCopySrc,
    dev_prelude::{
        FillExtent, ForEach, ForEachMut, ForEachMutPtr, Get, GetMut, GetMutPtr, GetRef, ReadExtent,
        TransformMap, WriteExtent,
    },
    multi_ptr::*,
    prelude::{GetMutUnchecked, GetRefUnchecked, GetUnchecked},
};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};
use core::ops::Add;
use either::Either;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A map from lattice location `PointN<N>` to data `T`, stored as a flat array.
///
/// See the [module-level docs](self) for more details.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Array<N, Chan> {
    channels: Chan,
    extent: ExtentN<N>,
}

macro_rules! array_n_type_alias {
    ($name:ident, $( $chan:ident : $store:ident ),+ ) => {
        pub type $name<N, $( $chan ),+, $( $store = Box<[$chan]> ),+> = Array<N, ($( Channel<$chan, $store> ),+)>;
    };
}

macro_rules! array_type_alias {
    ($name:ident, $dim:ty, $( $chan:ident : $store:ident ),+ ) => {
        pub type $name<$( $chan ),+, $( $store = Box<[$chan]> ),+> = Array<$dim, ($( Channel<$chan, $store> ),+)>;
    };
}

pub type ArrayNx1<N, A, S1 = Box<[A]>> = Array<N, Channel<A, S1>>;
array_n_type_alias!(ArrayNx2, A: S1, B: S2);
array_n_type_alias!(ArrayNx3, A: S1, B: S2, C: S3);
array_n_type_alias!(ArrayNx4, A: S1, B: S2, C: S3, D: S4);
array_n_type_alias!(ArrayNx5, A: S1, B: S2, C: S3, D: S4, E: S5);
array_n_type_alias!(ArrayNx6, A: S1, B: S2, C: S3, D: S4, E: S5, F: S6);

pub mod multichannel_aliases {
    use super::*;

    pub type Array2x1<A, S1 = Box<[A]>> = Array<[i32; 2], Channel<A, S1>>;
    array_type_alias!(Array2x2, [i32; 2], A: S1, B: S2);
    array_type_alias!(Array2x3, [i32; 2], A: S1, B: S2, C: S3);
    array_type_alias!(Array2x4, [i32; 2], A: S1, B: S2, C: S3, D: S4);
    array_type_alias!(Array2x5, [i32; 2], A: S1, B: S2, C: S3, D: S4, E: S5);
    array_type_alias!(Array2x6, [i32; 2], A: S1, B: S2, C: S3, D: S4, E: S5, F: S6);

    pub type Array3x1<A, S1 = Box<[A]>> = Array<[i32; 3], Channel<A, S1>>;
    array_type_alias!(Array3x2, [i32; 3], A: S1, B: S2);
    array_type_alias!(Array3x3, [i32; 3], A: S1, B: S2, C: S3);
    array_type_alias!(Array3x4, [i32; 3], A: S1, B: S2, C: S3, D: S4);
    array_type_alias!(Array3x5, [i32; 3], A: S1, B: S2, C: S3, D: S4, E: S5);
    array_type_alias!(Array3x6, [i32; 3], A: S1, B: S2, C: S3, D: S4, E: S5, F: S6);
}

pub use multichannel_aliases::*;

impl<N, Chan> Array<N, Chan> {
    /// Create a new `Array` directly from the extent and values.
    pub fn new(extent: ExtentN<N>, channels: Chan) -> Self {
        // TODO: assert that channels has length matching extent
        Self { channels, extent }
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

impl<N, Chan> Array<N, Chan>
where
    PointN<N>: IntegerPoint,
{
    /// Creates a new `Array` from the return value of `selector`. `selector` takes a tuple of `Channel`s that borrow their
    /// storage.
    #[inline]
    pub fn borrow_channels<'a, NewChan>(
        &'a self,
        selector: impl Fn(Chan::Borrowed) -> NewChan,
    ) -> Array<N, NewChan>
    where
        Chan: BorrowChannels<'a>,
    {
        Array::new(self.extent, selector(self.channels().borrow()))
    }

    /// Creates a new `Array` from the return value of `selector`. `selector` takes a tuple of `Channel`s that mutably borrow
    /// their storage.
    #[inline]
    pub fn borrow_channels_mut<'a, NewChan>(
        &'a mut self,
        selector: impl Fn(Chan::Borrowed) -> NewChan,
    ) -> Array<N, NewChan>
    where
        Chan: BorrowChannelsMut<'a>,
    {
        Array::new(self.extent, selector(self.channels_mut().borrow_mut()))
    }

    /// Sets the extent minimum to `p`. This doesn't change the shape of the extent.
    #[inline]
    pub fn set_minimum(&mut self, p: PointN<N>) {
        self.extent.minimum = p;
    }

    /// Adds `p` to the extent minimum.
    #[inline]
    pub fn translate(&mut self, p: PointN<N>) {
        self.extent = self.extent.add(p);
    }

    /// Returns `true` iff this map contains point `p`.
    #[inline]
    pub fn contains(&self, p: PointN<N>) -> bool {
        self.extent.contains(p)
    }
}

impl<N, Chan> FillExtent<N> for Array<N, Chan>
where
    Self: ForEachMutPtr<N, (), Item = Chan::Ptr>,
    PointN<N>: IntegerPoint,
    Chan: ResetChannels,
    Chan::Data: Clone,
{
    type Item = Chan::Data;

    /// Fill the entire `extent` with the same `value`.
    fn fill_extent(&mut self, extent: &ExtentN<N>, value: Self::Item) {
        if self.extent.eq(extent) {
            self.channels.reset_values(value);
        } else {
            unsafe {
                self.for_each_mut_ptr(extent, |_: (), v| v.write(value.clone()));
            }
        }
    }
}

impl<N, Chan> Array<N, Chan>
where
    Chan: ResetChannels,
{
    /// Set all points to the same value.
    #[inline]
    pub fn reset_values(&mut self, value: Chan::Data)
    where
        Chan::Data: Clone,
    {
        self.channels.reset_values(value);
    }
}

impl<N, Chan> Array<N, Chan>
where
    PointN<N>: IntegerPoint,
    Chan: FillChannels,
{
    /// Creates a map that fills the entire `extent` with the same `value`.
    pub fn fill(extent: ExtentN<N>, value: Chan::Data) -> Self
    where
        Chan::Data: Clone,
    {
        Self::new(extent, Chan::fill(extent.num_points(), value))
    }
}

impl<N, Chan, UninitChan> Array<N, Chan>
where
    Array<N, UninitChan>: ForEachMutPtr<N, PointN<N>, Item = UninitChan::Ptr>,
    PointN<N>: IntegerPoint,
    Chan: Channels<UninitSelf = UninitChan>,
    UninitChan: UninitChannels<InitSelf = Chan>,
    UninitChan::Ptr: IntoMultiMutPtr<Data = Chan::Data>,
{
    /// Create a new array for `extent` where each point's value is determined by the `filler` function.
    pub fn fill_with(extent: ExtentN<N>, mut filler: impl FnMut(PointN<N>) -> Chan::Data) -> Self {
        let mut array = unsafe { Array::<_, UninitChan>::maybe_uninit(extent) };
        unsafe {

            array.for_each_mut_ptr(&extent, |p, val| {
                val.into_multi_mut_ptr().write(filler(p));
            });

            array.assume_init()
        }
    }
}

impl<N, Chan> Array<N, Chan>
where
    PointN<N>: IntegerPoint,
    Chan: UninitChannels,
{
    /// Creates an uninitialized map, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    pub unsafe fn maybe_uninit(extent: ExtentN<N>) -> Array<N, Chan> {
        Array::new(extent, Chan::maybe_uninit(extent.num_points()))
    }

    /// Transmutes the map values in each channel from `MaybeUninit<T>` to `T` after manual initialization.
    /// # Safety
    /// All elements of the map must be initialized.
    pub unsafe fn assume_init(self) -> Array<N, Chan::InitSelf> {
        let (extent, channel) = self.into_parts();

        Array::new(extent, channel.assume_init())
    }
}

impl<N, T, Store> ArrayNx1<N, T, Store>
where
    PointN<N>: IntegerPoint,
    Store: AsRef<[T]>,
{
    pub fn new_one_channel(extent: ExtentN<N>, values: Store) -> Self {
        assert_eq!(extent.num_points(), values.as_ref().len());

        Self::new(extent, Channel::new(values))
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

impl<N, Chan> GetUnchecked<Stride> for Array<N, Chan>
where
    Chan: GetUnchecked<usize>,
{
    type Item = Chan::Item;

    #[inline]
    unsafe fn get_unchecked(&self, stride: Stride) -> Self::Item {
        self.channels.get_unchecked(stride.0)
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

impl<'a, N, Chan> GetRefUnchecked<'a, Stride> for Array<N, Chan>
where
    Chan: GetRefUnchecked<'a, usize>,
{
    type Item = Chan::Item;

    #[inline]
    unsafe fn get_ref_unchecked(&'a self, stride: Stride) -> Self::Item {
        self.channels.get_ref_unchecked(stride.0)
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

impl<'a, N, Chan> GetMutUnchecked<'a, Stride> for Array<N, Chan>
where
    Chan: GetMutUnchecked<'a, usize>,
{
    type Item = Chan::Item;

    #[inline]
    unsafe fn get_mut_unchecked(&'a mut self, stride: Stride) -> Self::Item {
        self.channels.get_mut_unchecked(stride.0)
    }
}

impl<N, Chan> GetMutPtr<Stride> for Array<N, Chan>
where
    Chan: GetMutPtr<usize>,
{
    type Item = Chan::Item;

    #[inline]
    unsafe fn get_mut_ptr(&mut self, stride: Stride) -> Self::Item {
        self.channels.get_mut_ptr(stride.0)
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

impl<N, Chan> GetUnchecked<Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetUnchecked<Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as GetUnchecked<Stride>>::Item;

    #[inline]
    unsafe fn get_unchecked(&self, p: Local<N>) -> Self::Item {
        self.get_unchecked(self.stride_from_local_point(p))
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

impl<'a, N, Chan> GetRefUnchecked<'a, Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetRefUnchecked<'a, Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as GetRefUnchecked<'a, Stride>>::Item;

    #[inline]
    unsafe fn get_ref_unchecked(&'a self, p: Local<N>) -> Self::Item {
        self.get_ref_unchecked(self.stride_from_local_point(p))
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

impl<'a, N, Chan> GetMutUnchecked<'a, Local<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetMutUnchecked<'a, Stride>,
    PointN<N>: Copy,
{
    type Item = <Self as GetMutUnchecked<'a, Stride>>::Item;

    #[inline]
    unsafe fn get_mut_unchecked(&'a mut self, p: Local<N>) -> Self::Item {
        self.get_mut_unchecked(self.stride_from_local_point(p))
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

impl<N, Chan> GetUnchecked<PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetUnchecked<Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as GetUnchecked<Local<N>>>::Item;

    #[inline]
    unsafe fn get_unchecked(&self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get_unchecked(Local(local_p))
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

impl<'a, N, Chan> GetRefUnchecked<'a, PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetRefUnchecked<'a, Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as GetRefUnchecked<'a, Local<N>>>::Item;

    #[inline]
    unsafe fn get_ref_unchecked(&'a self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get_ref_unchecked(Local(local_p))
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

impl<'a, N, Chan> GetMutUnchecked<'a, PointN<N>> for Array<N, Chan>
where
    Self: IndexedArray<N> + GetMutUnchecked<'a, Local<N>>,
    PointN<N>: Point,
{
    type Item = <Self as GetMutUnchecked<'a, Local<N>>>::Item;

    #[inline]
    unsafe fn get_mut_unchecked(&'a mut self, p: PointN<N>) -> Self::Item {
        let local_p = p - self.extent().minimum;

        self.get_mut_unchecked(Local(local_p))
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
        impl<N, Chan> ForEach<N, $coords> for Array<N, Chan>
        where
            Self: GetUnchecked<Stride>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint,
        {
            type Item = <Self as GetUnchecked<Stride>>::Item;

            #[inline]
            fn for_each(&self, iter_extent: &ExtentN<N>, mut f: impl FnMut($coords, Self::Item)) {
                let visitor = ArrayForEach::new_global(*self.extent(), *iter_extent);
                visitor.for_each(|$p, $stride| {
                    // This is safe because we guarantee that we don't access out of bounds.
                    f($forward_coords, unsafe { self.get_unchecked($stride) })
                });
            }
        }

        impl<'a, N, Chan> ForEachMutPtr<N, $coords> for Array<N, Chan>
        where
            Self: GetMutPtr<Stride, Item = Chan::Ptr>,
            N: ArrayIndexer<N>,
            PointN<N>: IntegerPoint,
            Chan: Channels,
        {
            type Item = Chan::Ptr;

            #[inline]
            unsafe fn for_each_mut_ptr(
                &mut self,
                iter_extent: &ExtentN<N>,
                mut f: impl FnMut($coords, Self::Item),
            ) {
                let visitor = ArrayForEach::new_global(*self.extent(), *iter_extent);
                visitor.for_each(|$p, $stride| {
                    f($forward_coords, self.get_mut_ptr($stride));
                });
            }
        }

        impl<'a, N, Chan> ForEachMut<'a, N, $coords> for Array<N, Chan>
        where
            Self: ForEachMutPtr<N, $coords, Item = Chan::Ptr>,
            Chan: Channels,
            Chan::Ptr: IntoMultiMut<'a>,
        {
            type Item = <Chan::Ptr as IntoMultiMut<'a>>::MultiMut;

            #[inline]
            fn for_each_mut(
                &'a mut self,
                iter_extent: &ExtentN<N>,
                mut f: impl FnMut($coords, Self::Item),
            ) {
                unsafe {
                    self.for_each_mut_ptr(iter_extent, |c, ptr| f(c, ptr.into_multi_mut()));
                }
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

impl<'a, N: 'a, Chan: 'a> ReadExtent<'a, N> for Array<N, Chan>
where
    PointN<N>: IntegerPoint,
{
    type Src = ArrayCopySrc<&'a Array<N, Chan>>;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        let in_bounds_extent = extent.intersection(self.extent());

        once((in_bounds_extent, ArrayCopySrc(self)))
    }
}

impl<'a, N, Data, SrcSlices, ChanSrc, ChanDst> WriteExtent<N, ArrayCopySrc<&'a Array<N, ChanSrc>>>
    for Array<N, ChanDst>
where
    Self: GetMutPtr<Stride, Item = ChanDst::Ptr>,
    Array<N, ChanSrc>: Get<Stride, Item = Data>,
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint,
    ChanSrc: Channels<Data = Data> + Slices<'a, Target = SrcSlices>,
    ChanDst: Channels<Data = Data> + CopySlices<'a, Src = SrcSlices>,
{
    fn write_extent(
        &mut self,
        extent: &ExtentN<N>,
        src_array: ArrayCopySrc<&'a Array<N, ChanSrc>>,
    ) {
        // It is assumed by the interface that extent is a subset of the src array, so we only need to intersect with the
        // destination.
        let in_bounds_extent = extent.intersection(self.extent());

        let copy_entire_array = in_bounds_extent.shape == self.extent().shape
            && in_bounds_extent.shape == src_array.0.extent().shape;

        if copy_entire_array {
            // Fast path, mostly for copying entire chunks between chunk maps.
            self.channels.copy_slices(src_array.0.channels.slices());
        } else {
            unchecked_copy_extent_between_arrays(self, src_array.0, in_bounds_extent);
        }
    }
}

impl<'a, N, Chan, Delegate, F> WriteExtent<N, ArrayCopySrc<TransformMap<'a, Delegate, F>>>
    for Array<N, Chan>
where
    Self: IndexedArray<N> + GetMutPtr<Stride, Item = Chan::Ptr>,
    TransformMap<'a, Delegate, F>: IndexedArray<N> + Get<Stride, Item = Chan::Data>,
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint,
    Chan: Channels,
{
    fn write_extent(
        &mut self,
        extent: &ExtentN<N>,
        src_array: ArrayCopySrc<TransformMap<'a, Delegate, F>>,
    ) {
        // It is assumed by the interface that extent is a subset of the src array, so we only need to intersect with the
        // destination.
        let in_bounds_extent = extent.intersection(self.extent());

        unchecked_copy_extent_between_arrays(self, &src_array.0, in_bounds_extent);
    }
}

// SAFETY: `extent` must be in-bounds of both arrays.
fn unchecked_copy_extent_between_arrays<Dst, Src, N, Ptr>(
    dst: &mut Dst,
    src: &Src,
    extent: ExtentN<N>,
) where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint,
    Dst: IndexedArray<N> + GetMutPtr<Stride, Item = Ptr>,
    Src: IndexedArray<N> + Get<Stride, Item = Ptr::Data>,
    Ptr: MultiMutPtr,
{
    // It shoudn't matter which type we use for the indexer.
    let for_each = LockStepArrayForEach::new_global_unchecked(extent, *dst.extent(), *src.extent());
    Dst::Indexer::for_each_lockstep_unchecked(for_each, |_p, (s_dst, s_src)| {
        // The actual copy.
        // PERF: could be faster with SIMD copy
        unsafe {
            dst.get_mut_ptr(s_dst).write(src.get(s_src));
        }
    });
}

impl<N, Chan, Ch> WriteExtent<N, ChunkCopySrc<N, Chan::Data, Ch>> for Array<N, Chan>
where
    Self: ForEachMutPtr<N, (), Item = Chan::Ptr> + WriteExtent<N, ArrayCopySrc<Ch>>,
    PointN<N>: IntegerPoint,
    Chan: ResetChannels,
    Chan::Data: Clone,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: ChunkCopySrc<N, Chan::Data, Ch>) {
        match src {
            Either::Left(array) => self.write_extent(extent, array),
            Either::Right(ambient) => self.fill_extent(extent, ambient.get()),
        }
    }
}

impl<N, Chan, F> WriteExtent<N, F> for Array<N, Chan>
where
    Self: ForEachMutPtr<N, PointN<N>, Item = Chan::Ptr>,
    F: Fn(PointN<N>) -> Chan::Data,
    Chan: Channels,
{
    fn write_extent(&mut self, extent: &ExtentN<N>, src: F) {
        unsafe {
            self.for_each_mut_ptr(extent, |p, v| v.write((src)(p)));
        }
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
    use crate::prelude::{copy_extent, Array2x1, Array3x1};
    use core::mem::MaybeUninit;

    #[test]
    fn fill_and_get_2d() {
        let extent = Extent2i::from_min_and_shape(PointN([1, 1]), PointN([10, 10]));
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
        let extent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
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
        let extent = Extent2i::from_min_and_shape(Point2i::fill(1), Point2i::fill(10));
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
        let extent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
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
        let extent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(10));
        let mut array: Array3x1<MaybeUninit<i32>> = unsafe { Array3x1::maybe_uninit(extent) };

        array.for_each_mut(&extent, |_: (), val| unsafe {
            val.as_mut_ptr().write(1);
        });

        let array = unsafe { array.assume_init() };

        array.for_each(&extent, |_: (), val| {
            assert_eq!(val, 1i32);
        });
    }

    #[test]
    fn copy() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x1::fill(extent, 0);

        let subextent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(5));
        array.for_each_mut(&subextent, |p: Point3i, val| {
            *val = p.x() + p.y() + p.z();
        });

        let mut other_array = Array3x1::fill(extent, 0);
        copy_extent(&subextent, &array, &mut other_array);

        assert_eq!(array, other_array);
    }

    #[test]
    fn multichannel_get() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x2::fill(extent, (0, 'a'));

        assert_eq!(array.get(Stride(0)), (0, 'a'));
        assert_eq!(array.get_ref(Stride(0)), (&0, &'a'));
        assert_eq!(array.get_mut(Stride(0)), (&mut 0, &mut 'a'));

        assert_eq!(array.get(Local(Point3i::fill(0))), (0, 'a'));
        assert_eq!(array.get_ref(Local(Point3i::fill(0))), (&0, &'a'));
        assert_eq!(array.get_mut(Local(Point3i::fill(0))), (&mut 0, &mut 'a'));

        assert_eq!(array.get(Point3i::fill(0)), (0, 'a'));
        assert_eq!(array.get_ref(Point3i::fill(0)), (&0, &'a'));
        assert_eq!(array.get_mut(Point3i::fill(0)), (&mut 0, &mut 'a'));
    }

    #[test]
    fn multichannel_access_with_borrowed_storage() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x2::fill(extent, (0, 'a'));

        // Swap the order.
        let borrowed = array.borrow_channels(|(c1, c2)| (c2, c1));

        assert_eq!(borrowed.get(Stride(0)), ('a', 0));
        assert_eq!(borrowed.get_ref(Stride(0)), (&'a', &0));

        assert_eq!(borrowed.get(Local(Point3i::fill(0))), ('a', 0));
        assert_eq!(borrowed.get_ref(Local(Point3i::fill(0))), (&'a', &0));

        borrowed.for_each(&extent, |_: (), x| assert_eq!(x, ('a', 0)));

        let mut other = Array3x2::fill(extent, ('b', 1));
        copy_extent(&extent, &borrowed, &mut other);

        let mut borrowed = array.borrow_channels_mut(|(c1, c2)| (c2, c1));

        assert_eq!(borrowed.get(Stride(0)), ('a', 0));
        assert_eq!(borrowed.get_ref(Stride(0)), (&'a', &0));
        assert_eq!(borrowed.get_mut(Stride(0)), (&mut 'a', &mut 0));

        assert_eq!(borrowed.get(Local(Point3i::fill(0))), ('a', 0));
        assert_eq!(borrowed.get_ref(Local(Point3i::fill(0))), (&'a', &0));
        assert_eq!(
            borrowed.get_mut(Local(Point3i::fill(0))),
            (&mut 'a', &mut 0)
        );

        borrowed.for_each_mut(&extent, |_: (), x| assert_eq!(x, (&mut 'a', &mut 0)));

        copy_extent(&extent, &other, &mut borrowed);
    }

    #[test]
    fn multichannel_for_each() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x2::fill(extent, (0, 'a'));

        array.for_each(&extent, |_: (), (c1, c2)| {
            assert_eq!(c1, 0);
            assert_eq!(c2, 'a');
        });

        array.for_each_mut(&extent, |_: (), (c1, c2)| {
            *c1 = 1;
            *c2 = 'b';
        });

        array.for_each(&extent, |_: (), (c1, c2)| {
            assert_eq!(c1, 1);
            assert_eq!(c2, 'b');
        });
    }

    #[test]
    fn multichannel_fill_extent() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let mut array = Array3x2::fill(extent, (0, 'a'));

        array.fill_extent(&extent, (1, 'b'));

        array.for_each(array.extent(), |_: (), (num, letter)| {
            assert_eq!(num, 1);
            assert_eq!(letter, 'b');
        });
    }

    #[test]
    fn multichannel_fill_with() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let array =
            Array3x2::fill_with(extent, |p| if p.x() % 2 == 0 { (1, 'b') } else { (0, 'a') });

        array.for_each(array.extent(), |p: Point3i, (num, letter)| {
            if p.x() % 2 == 0 {
                assert_eq!((num, letter), (1, 'b'));
            } else {
                assert_eq!((num, letter), (0, 'a'));
            }
        });
    }

    #[test]
    fn multichannel_copy() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let src = Array3x2::fill(extent, (0, 'a'));
        let mut dst = Array3x2::fill(extent, (1, 'b'));
        copy_extent(&extent, &src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn select_channel_with_transform() {
        let extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(10));
        let src = Array3x2::fill(extent, (0, 'a'));
        let src_select = TransformMap::new(&src, |(_num, letter): (i32, char)| letter);

        let mut dst = Array3x1::fill(extent, 'b');

        copy_extent(&extent, &src_select, &mut dst);

        dst.for_each(&extent, |_: (), letter| {
            assert_eq!(letter, 'a');
        });
    }
}
