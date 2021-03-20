use core::hash::{BuildHasher, Hash};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::pin::Pin;

/// A cache with a very specific niche. When reading from shared, two-tier storage, if you miss the cache and need to fetch from
/// the cold tier, then you also need a place to store the fetched data. Rather than doing interior mutation of the storage,
/// which requires synchronization, the fetched data can be stored in a thread-local cache, the `LocalCache`.
///
/// SAFE: We guarantee in these APIs that all references returned are valid for the lifetime of the `LocalCache`, even as new
/// values are added to the map. The invariants are:
///   1. Once a value is placed here, it will never get dropped or moved until calling `into_iter`.
///   2. The values are placed into `Pin<Box<V>>` so the memory address is guaranteed stable.
///   3. Returned references must be dropped before calling `into_iter`.
#[derive(Default)]
pub struct LocalCache<K, V, H> {
    store: UnsafeCell<HashMap<K, Pin<Box<V>>, H>>,
}

impl<K, V, H> LocalCache<K, V, H>
where
    K: Eq + Hash,
    H: Default + BuildHasher,
{
    pub fn new() -> Self {
        LocalCache {
            store: UnsafeCell::new(HashMap::with_hasher(Default::default())),
        }
    }

    /// Fetch the value for `key`. If it's not here, call `f` to fetch it.
    pub fn get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> &V {
        let mut_store = unsafe { &mut *self.store.get() };

        mut_store.entry(key).or_insert_with(|| Box::pin(f()))
    }

    // TODO: impl IntoIterator instead
    /// Consume self and iterate over all (key, value) pairs.
    pub fn flush_iter(self) -> impl Iterator<Item = (K, V)> {
        self.store
            .into_inner()
            .into_iter()
            .map(|(k, v)| (k, unsafe { *Pin::into_inner_unchecked(v) }))
    }
}
