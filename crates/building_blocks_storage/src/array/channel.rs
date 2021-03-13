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

impl_get_via_get_ref_and_clone!(Channel<T, Store>, T, Store);

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    use crate::Get;

    #[test]
    fn tuple_of_channels_can_get() {
        let mut ch1 = Channel::new_fill(0, 10);
        let mut ch2 = Channel::new_fill(0, 10);

        assert_eq!((&ch1, &ch2).get(0), (0, 0));
        assert_eq!((&ch1, &ch2).get_ref(0), (&0, &0));
        assert_eq!((&mut ch1, &mut ch2).get_mut(0), (&mut 0, &mut 0));

        let mut owned = (ch1, ch2);

        assert_eq!(owned.get(0), (0, 0));
        assert_eq!(owned.get_ref(0), (&0, &0));
        assert_eq!(owned.get_mut(0), (&mut 0, &mut 0));
    }
}
