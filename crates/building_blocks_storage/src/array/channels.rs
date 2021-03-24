pub mod channel;
pub mod compression;
pub mod multichannel;

pub use channel::*;
pub use compression::*;
pub use multichannel::*;

use crate::MultiMutPtr;

pub trait Channels {
    type Data;
    type Ptr: MultiMutPtr<Data = Self::Data>;
    type UninitSelf: UninitChannels;
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
