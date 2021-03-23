use crate::{AsRawBytes, Channels, FillChannels, GetMut, GetMutPtr, GetRef, UninitChannels};

use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Channel<T, Store = Vec<T>> {
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

impl<T> Channel<T, Vec<T>> {
    pub fn fill(value: T, length: usize) -> Self
    where
        T: Clone,
    {
        Self::new(vec![value; length])
    }
}

impl<T, Store> Channel<T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    #[inline]
    pub fn reset_values(&mut self, value: T)
    where
        T: Clone,
    {
        self.store.fill(value);
    }
}

impl<'a, T, Store> AsRawBytes<'a> for Channel<T, Store>
where
    T: 'static + Copy,
    Store: Deref<Target = [T]>,
{
    type Output = &'a [u8];

    fn as_raw_bytes(&'a self) -> Self::Output {
        self.store().as_raw_bytes()
    }
}

impl<T> Channels for Channel<T> {
    type Data = T;
    type Ptr = *mut T;
    type UninitSelf = Channel<MaybeUninit<T>>;
}

impl<T> FillChannels for Channel<T>
where
    T: Clone,
{
    fn fill(value: Self::Data, length: usize) -> Self {
        Self::fill(value, length)
    }

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
        let mut store = Vec::with_capacity(size);
        store.set_len(size);

        Channel::new(store)
    }

    /// Transmutes the channel values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just
    /// reconstructs the internal `Vec` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the channel must be initialized.
    unsafe fn assume_init(self) -> Self::InitSelf {
        let transmuted_values = {
            // Ensure the original vector is not dropped.
            let mut v_clone = core::mem::ManuallyDrop::new(self.store);

            Vec::from_raw_parts(
                v_clone.as_mut_ptr() as *mut T,
                v_clone.len(),
                v_clone.capacity(),
            )
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

impl<'a, T, Store> GetRef<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: Deref<Target = [T]>,
{
    type Item = &'a T;

    #[inline]
    fn get_ref(&'a self, offset: usize) -> Self::Item {
        if cfg!(debug_assertions) {
            &self.store[offset]
        } else {
            unsafe { self.store.get_unchecked(offset) }
        }
    }
}

impl<'a, T, Store> GetMut<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: DerefMut<Target = [T]>,
{
    type Item = &'a mut T;

    #[inline]
    fn get_mut(&'a mut self, offset: usize) -> Self::Item {
        if cfg!(debug_assertions) {
            &mut self.store[offset]
        } else {
            unsafe { self.store.get_unchecked_mut(offset) }
        }
    }
}

impl<T, Store> GetMutPtr<usize> for Channel<T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    type Item = *mut T;

    #[inline]
    unsafe fn get_mut_ptr(&mut self, offset: usize) -> Self::Item {
        self.store.as_mut_ptr().add(offset)
    }
}

impl_get_via_get_ref_and_clone!(Channel<T, Store>, T, Store);
