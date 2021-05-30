pub mod channel;
pub mod compression;
pub mod multichannel;

pub use channel::*;
pub use compression::*;
pub use multichannel::*;

use crate::MultiMutPtr;

/// Implemented by any tuple of `Channel`s to indicate the types of data being stored.
pub trait Channels {
    type Data;
    type Ptr: MultiMutPtr<Data = Self::Data>;
    type UninitSelf: UninitChannels;
}

/// Converts a tuple of channels into a tuple of slices.
pub trait Slices<'a> {
    type Target;

    fn slices(&'a self) -> Self::Target;
}

/// Converts a tuple of channels into a tuple of mutable slices.
pub trait SlicesMut<'a> {
    type Target;

    fn slices_mut(&'a mut self) -> Self::Target;
}

pub trait CopySlices<'a> {
    type Src;

    fn copy_slices(&mut self, src: Self::Src);
}

/// Converts a tuple of channels that own their data into a tuple of channels that borrow their data.
pub trait BorrowChannels<'a> {
    type Borrowed;

    fn borrow(&'a self) -> Self::Borrowed;
}

/// Converts a tuple of channels that own their data into a tuple of channels that mutably borrow their data.
pub trait BorrowChannelsMut<'a> {
    type Borrowed;

    fn borrow_mut(&'a mut self) -> Self::Borrowed;
}

pub trait FillChannels: Channels {
    fn fill(value: Self::Data, length: usize) -> Self;
    fn reset_values(&mut self, value: Self::Data);
}

pub trait UninitChannels: Channels {
    type InitSelf;

    /// # Safety
    /// Elements should not be read until they are initialized.
    unsafe fn maybe_uninit(size: usize) -> Self;

    /// # Safety
    /// All elements of the channel must be initialized.
    unsafe fn assume_init(self) -> Self::InitSelf;
}
