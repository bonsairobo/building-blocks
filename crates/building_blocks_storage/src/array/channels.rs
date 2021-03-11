use core::ops::{Deref, DerefMut};

pub struct Channel<T, Store> {
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
