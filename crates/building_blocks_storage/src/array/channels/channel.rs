use crate::{
    array::{
        BorrowChannels, BorrowChannelsMut, CopySlices, FillChannels, ResetChannels, Slices,
        SlicesMut, UninitChannels,
    },
    dev_prelude::{Channels, GetMut, GetMutPtr, GetRef},
    prelude::{GetMutUnchecked, GetRefUnchecked},
};

use core::mem::MaybeUninit;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Channel<T, Store = Box<[T]>> {
    store: Store,
    marker: std::marker::PhantomData<T>,
}

impl<T, Store> Channel<T, Store> {
    #[inline]
    pub fn new(store: Store) -> Self {
        Self {
            store,
            marker: Default::default(),
        }
    }

    #[inline]
    pub fn take_store(self) -> Store {
        self.store
    }

    #[inline]
    pub fn store(&self) -> &Store {
        &self.store
    }

    #[inline]
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }
}

impl<T> Channel<T> {
    pub fn fill(length: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::new(vec![value; length].into_boxed_slice())
    }

    pub fn fill_with(length: usize, mut filler: impl FnMut() -> T) -> Self {
        let mut chan = unsafe { Channel::maybe_uninit(length) };
        chan.store.fill_with(|| MaybeUninit::new(filler()));
        unsafe { chan.assume_init() }
    }
}

impl<T, Store> Channel<T, Store>
where
    Store: AsMut<[T]>,
{
    #[inline]
    pub fn reset_values(&mut self, value: T)
    where
        T: Clone,
    {
        self.store.as_mut().fill(value);
    }
}

impl<T, Store> Channels for Channel<T, Store> {
    type Data = T;
    type Ptr = *mut T;
    type UninitSelf = Channel<MaybeUninit<T>>;
}

impl<'a, T: 'a, Store> Slices<'a> for Channel<T, Store>
where
    Store: AsRef<[T]>,
{
    type Target = &'a [T];

    fn slices(&'a self) -> Self::Target {
        self.store.as_ref()
    }
}

impl<'a, T: 'a, Store> SlicesMut<'a> for Channel<T, Store>
where
    Store: AsMut<[T]>,
{
    type Target = &'a [T];

    fn slices_mut(&'a mut self) -> Self::Target {
        self.store.as_mut()
    }
}

impl<'a, T: 'a, Store> CopySlices<'a> for Channel<T, Store>
where
    T: Clone,
    Store: AsMut<[T]>,
{
    type Src = &'a [T];

    fn copy_slices(&mut self, src: Self::Src) {
        self.store.as_mut().clone_from_slice(src)
    }
}

impl<'a, T: 'a, Store> BorrowChannels<'a> for Channel<T, Store>
where
    Store: AsRef<[T]>,
{
    type Borrowed = Channel<T, &'a [T]>;

    fn borrow(&'a self) -> Self::Borrowed {
        Channel::new(self.store.as_ref())
    }
}

impl<'a, T: 'a, Store> BorrowChannelsMut<'a> for Channel<T, Store>
where
    Store: AsMut<[T]>,
{
    type Borrowed = Channel<T, &'a mut [T]>;

    fn borrow_mut(&'a mut self) -> Self::Borrowed {
        Channel::new(self.store.as_mut())
    }
}

impl<T> FillChannels for Channel<T>
where
    T: Clone,
{
    fn fill(length: usize, value: Self::Data) -> Self {
        Self::fill(length, value)
    }
}

impl<T, Store> ResetChannels for Channel<T, Store>
where
    T: Clone,
    Store: AsMut<[T]>,
{
    fn reset_values(&mut self, value: Self::Data) {
        self.reset_values(value)
    }
}

impl<T> UninitChannels for Channel<MaybeUninit<T>> {
    type InitSelf = Channel<T>;

    /// Creates an uninitialized channel, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    unsafe fn maybe_uninit(size: usize) -> Self {
        // TODO: use Box::new_uninit_slice when it's stable
        let mut store = Vec::with_capacity(size);
        store.set_len(size);
        Channel::new(store.into_boxed_slice())
    }

    /// Transmutes the channel values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just
    /// reconstructs the internal `Box<[T]>` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the channel must be initialized.
    unsafe fn assume_init(self) -> Self::InitSelf {
        let transmuted_values = {
            // Ensure the original slice is not dropped.
            let mut v_clone = core::mem::ManuallyDrop::new(self.store);
            Box::from_raw(std::slice::from_raw_parts_mut(
                v_clone.as_mut_ptr() as *mut T,
                v_clone.len(),
            ))
        };

        Channel::new(transmuted_values)
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, T, Store> GetRefUnchecked<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: AsRef<[T]>,
{
    type Item = &'a T;

    #[inline]
    unsafe fn get_ref_unchecked(&'a self, offset: usize) -> Self::Item {
        self.store.as_ref().get_unchecked(offset)
    }
}

impl<'a, T, Store> GetRef<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: AsRef<[T]>,
{
    type Item = &'a T;

    #[inline]
    fn get_ref(&'a self, offset: usize) -> Self::Item {
        &self.store.as_ref()[offset]
    }
}

impl<'a, T, Store> GetMutUnchecked<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: AsMut<[T]>,
{
    type Item = &'a mut T;

    #[inline]
    unsafe fn get_mut_unchecked(&'a mut self, offset: usize) -> Self::Item {
        self.store.as_mut().get_unchecked_mut(offset)
    }
}

impl<'a, T, Store> GetMut<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: AsMut<[T]>,
{
    type Item = &'a mut T;

    #[inline]
    fn get_mut(&'a mut self, offset: usize) -> Self::Item {
        &mut self.store.as_mut()[offset]
    }
}

impl<T, Store> GetMutPtr<usize> for Channel<T, Store>
where
    Store: AsMut<[T]>,
{
    type Item = *mut T;

    #[inline]
    unsafe fn get_mut_ptr(&mut self, offset: usize) -> Self::Item {
        self.store.as_mut().as_mut_ptr().add(offset)
    }
}

impl_get_via_get_ref_and_clone!(Channel<T, Store>, T, Store);
