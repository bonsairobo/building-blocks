use core::hash::{BuildHasher, Hash};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::pin::Pin;

/// A cache with a very specific niche. When reading from shared, two-tier storage, if you miss the cache and need to fetch from
/// the cold tier, then you also need a place to store the fetched data. Rather than doing interior mutation of the storage,
/// which requires synchronization, the fetched data can be stored in a thread-local cache, the `LocalCache`.
///
/// # Safety
///
/// We guarantee in these APIs that all references returned are valid for the lifetime of the `LocalCache`, even as new values
/// are added to the map. The invariants are:
///   1. Once a value is placed here, it will never get dropped or moved until calling `drain_iter`.
///   2. The values are placed into `Pin<Box<V>>` so the memory address is guaranteed stable.
///   3. Returned references must be dropped before calling `drain_iter` (since it borrows self mutably).
pub struct LocalCache<K, V, H> {
    store: UnsafeCell<HashMap<K, Pin<Box<V>>, H>>,
}

impl<K, V, H> Default for LocalCache<K, V, H>
where
    H: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, H> LocalCache<K, V, H>
where
    H: Default,
{
    pub fn new() -> Self {
        LocalCache {
            store: UnsafeCell::new(HashMap::with_hasher(Default::default())),
        }
    }
}

impl<K, V, H> LocalCache<K, V, H>
where
    K: Eq + Hash,
    H: Default + BuildHasher,
{
    /// Fetch the value for `key`.
    pub fn get(&self, key: K) -> Option<&V> {
        let store = unsafe { &*self.store.get() };
        store.get(&key).map(|v| &**v)
    }

    /// Fetch the value for `key`. If it's not here, call `f` to fetch it.
    pub fn get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> &V {
        let mut_store = unsafe { &mut *self.store.get() };

        mut_store.entry(key).or_insert_with(|| Box::pin(f()))
    }

    /// Consume and iterate over all (key, value) pairs.
    pub fn drain_iter(&mut self) -> impl '_ + Iterator<Item = (K, V)> {
        self.store
            .get_mut()
            .drain()
            .map(|(k, v)| (k, unsafe { *Pin::into_inner_unchecked(v) }))
    }
}
