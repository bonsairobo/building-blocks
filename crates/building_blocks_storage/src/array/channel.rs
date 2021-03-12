use crate::{GetMut, GetRef};

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
    pub fn new_fill(value: T, length: usize) -> Self
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
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.store.fill(value);
    }
}

impl<T> Channel<MaybeUninit<T>, Vec<MaybeUninit<T>>> {
    /// Creates an uninitialized channel, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    pub unsafe fn maybe_uninit(size: usize) -> Self {
        let mut store = Vec::with_capacity(size);
        store.set_len(size);

        Self::new(store)
    }

    /// Transmutes the channel values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just
    /// reconstructs the internal `Vec` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the map must be initialized.
    pub unsafe fn assume_init(self) -> Channel<T> {
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

impl<T, Store> GetRef<usize, T> for Channel<T, Store>
where
    Store: Deref<Target = [T]>,
{
    #[inline]
    fn get_ref(&self, offset: usize) -> &T {
        if cfg!(debug_assertions) {
            &self.store[offset]
        } else {
            unsafe { self.store.get_unchecked(offset) }
        }
    }
}

impl<T, Store> GetMut<usize, T> for Channel<T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    #[inline]
    fn get_mut(&mut self, offset: usize) -> &mut T {
        if cfg!(debug_assertions) {
            &mut self.store[offset]
        } else {
            unsafe { self.store.get_unchecked_mut(offset) }
        }
    }
}
