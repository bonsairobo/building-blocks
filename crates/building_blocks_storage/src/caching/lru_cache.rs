use crate::SmallKeyBuildHasher;

use core::hash::{BuildHasher, Hash};
use std::collections::{hash_map, HashMap};

/// A cache that tracks the Least Recently Used element for next eviction.
///
/// For the purpose of fast, repeated random access, LRU order is only updated on insertion or by calling "touch_if_cached."
///
/// Eviction does not happen inline; the user must explicitly call `evict_lru` to evict the LRU element. Thus the cache may grow
/// unbounded unless evictions or explicit removals occur.
///
/// Note that eviction and removal are not treated the same. The cache remembers elements that have been evicted but not
/// removed. This is useful when users need to store evicted data in a separate structure, since if they look up a key and get
/// `Some(CacheEntry::Evicted)`, they know that the data exists somewhere else. If they get `None`, then they don't have to look
/// elsewhere; the data simply doesn't exist anywhere.
#[derive(Clone, Debug)]
pub struct LruCache<K, V, E, H> {
    store: HashMap<K, CacheEntry<(V, usize), E>, H>,
    order: LruList<K>,
    num_evicted: usize,
}

/// An `LruCache` using the FNV hashing algorithm.
pub type SmallKeyLruCache<K, V, E = ()> = LruCache<K, V, E, SmallKeyBuildHasher>;

impl<K, V, E, H> Default for LruCache<K, V, E, H>
where
    H: Default,
    K: Hash + Eq,
{
    fn default() -> Self {
        Self::with_hasher(Default::default())
    }
}

impl<K, V, E, H> LruCache<K, V, E, H>
where
    K: Hash + Eq,
{
    pub fn with_hasher(hasher_builder: H) -> LruCache<K, V, E, H> {
        LruCache {
            store: HashMap::with_hasher(hasher_builder),
            order: LruList::new(),
            num_evicted: 0,
        }
    }
}

impl<K, V, E, H> LruCache<K, V, E, H>
where
    K: Hash + Eq + Clone,
    E: Copy,
    H: BuildHasher,
{
    /// Borrow the entry for `key`. This will not update the LRU order.
    #[inline]
    pub fn get(&self, key: &K) -> Option<CacheEntry<&V, E>> {
        self.store
            .get(key)
            .map(|entry| entry.as_ref().map_cached(|(val, _)| val))
    }

    /// Mutably borrow the entry for `key`. This will not update the LRU order.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<CacheEntry<&mut V, E>> {
        self.store
            .get_mut(key)
            .map(|entry| entry.as_mut().map_cached(|(val, _)| val))
    }

    /// Inserts a `new_val` for `key`, returning the old entry if it exists. `key` becomes the most recently used.
    #[inline]
    pub fn insert(&mut self, key: K, new_val: V) -> Option<CacheEntry<V, E>> {
        let Self { store, order, .. } = self;
        match store.entry(key.clone()) {
            hash_map::Entry::Occupied(occupied) => match occupied.into_mut() {
                CacheEntry::Cached((old_val, i)) => {
                    order.move_to_front(*i);

                    Some(CacheEntry::Cached(std::mem::replace(old_val, new_val)))
                }
                x => {
                    let new_i = order.push_front(Some(key));
                    let old_entry = std::mem::replace(x, CacheEntry::Cached((new_val, new_i)));
                    self.num_evicted -= 1;

                    Some(CacheEntry::Evicted(old_entry.unwrap_evicted()))
                }
            },
            hash_map::Entry::Vacant(vacant) => {
                let new_i = order.push_front(Some(key));
                vacant.insert(CacheEntry::Cached((new_val, new_i)));

                None
            }
        }
    }

    /// Marks `key` as most recently used if the entry is cached. Returns `false` iff the entry is evicted.
    ///
    /// This is useful to do before prefetching many values, since it tells you which values need to be fetched and otherwise
    /// updates the LRU order for the values that didn't need fetching (but were nonetheless desired).
    #[inline]
    pub fn touch_if_cached(&mut self, key: K) -> bool {
        let Self { store, order, .. } = self;

        if let Some(entry) = store.get(&key) {
            match entry {
                CacheEntry::Cached((_, i)) => {
                    order.move_to_front(*i);

                    true
                }
                CacheEntry::Evicted(_) => false,
            }
        } else {
            true
        }
    }

    /// Tries to get the value for `key`, returning it if it exists. If the entry state is evicted, calls `on_evicted` to
    /// repopulate the entry and marks it as most recently used. Otherwise, returns `None`.
    #[inline]
    pub fn get_mut_or_repopulate_with(
        &mut self,
        key: K,
        on_evicted: impl FnOnce(E) -> V,
    ) -> Option<&mut V> {
        let Self { store, order, .. } = self;
        match store.entry(key.clone()) {
            hash_map::Entry::Occupied(occupied) => match occupied.into_mut() {
                CacheEntry::Cached((val, _)) => Some(val),
                x => {
                    let repop_val = on_evicted(x.unwrap_evicted());
                    let new_i = order.push_front(Some(key));
                    std::mem::swap(x, &mut CacheEntry::Cached((repop_val, new_i)));
                    self.num_evicted -= 1;

                    Some(x.as_mut().unwrap_value())
                }
            },
            hash_map::Entry::Vacant(_) => None,
        }
    }

    /// Tries to get the value for `key`, returning it if it exists. If the entry state is evicted, calls `on_evicted` to
    /// repopulate the entry. If there is no entry, calls `on_missing` to populate the entry. In either case, if a new entry is
    /// created, it is marked as most recently used.
    #[inline]
    pub fn get_mut_or_insert_with(
        &mut self,
        key: K,
        on_evicted: impl FnOnce(E) -> V,
        on_missing: impl FnOnce() -> V,
    ) -> &mut V {
        let Self { store, order, .. } = self;
        match store.entry(key.clone()) {
            hash_map::Entry::Occupied(occupied) => match occupied.into_mut() {
                CacheEntry::Cached((val, _)) => val,
                x => {
                    let repop_val = on_evicted(x.unwrap_evicted());
                    let new_i = order.push_front(Some(key));
                    std::mem::swap(x, &mut CacheEntry::Cached((repop_val, new_i)));
                    self.num_evicted -= 1;

                    x.as_mut().unwrap_value()
                }
            },
            hash_map::Entry::Vacant(vacant) => {
                let new_val = on_missing();
                let new_i = order.push_front(Some(key));

                vacant
                    .insert(CacheEntry::Cached((new_val, new_i)))
                    .as_mut()
                    .unwrap_value()
            }
        }
    }

    /// Removes any trace of `key`, such that further accesses will return `None` until a new value is inserted.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<CacheEntry<V, E>> {
        self.store.remove(key).map(|entry| match entry {
            CacheEntry::Cached((val, i)) => {
                self.order.remove(i);

                CacheEntry::Cached(val)
            }
            CacheEntry::Evicted(location) => {
                self.num_evicted -= 1;

                CacheEntry::Evicted(location)
            }
        })
    }

    /// Evicts a specific `key`. This will leave a sentinel behind so that further accesses will return
    /// `Some(CacheEntry::Evicted(new_location))` until the key is removed or a new entry is inserted. If the entry is already
    /// evicted, this has no effect and just returns the current entry.
    #[inline]
    pub fn evict(&mut self, key: K, new_location: E) -> Option<CacheEntry<V, E>> {
        self.num_evicted += 1;
        self.store
            .insert(key, CacheEntry::Evicted(new_location))
            .map(|entry| match entry {
                CacheEntry::Cached((val, i)) => {
                    self.order.remove(i);

                    CacheEntry::Cached(val)
                }
                CacheEntry::Evicted(location) => {
                    // This is the only case where # evicted should not increase.
                    self.num_evicted -= 1;

                    CacheEntry::Evicted(location)
                }
            })
    }

    /// Evicts the least-recently used value. This will leave a sentinel behind so that further accesses will return
    /// `Some(CacheEntry::Evicted)` until the key is removed or a new entry is inserted.
    ///
    /// Nothing happens if the cache is empty.
    #[inline]
    pub fn evict_lru(&mut self, new_location: E) -> Option<(K, V)> {
        if self.len_cached() == 0 {
            return None;
        }

        let key = self.order.pop_back();
        let entry = std::mem::replace(
            self.store.get_mut(&key).unwrap(),
            CacheEntry::Evicted(new_location),
        );
        self.num_evicted += 1;

        Some((key, entry.unwrap_value()))
    }

    /// Removes the least-recently used value, leaving no trace.
    #[inline]
    pub fn remove_lru(&mut self) -> Option<(K, V)> {
        if self.len_cached() == 0 {
            return None;
        }

        let key = self.order.pop_back();
        let val = self.store.remove(&key).unwrap().unwrap_value();

        Some((key, val))
    }

    /// Delete all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.store.clear();
        self.order.clear();
        self.num_evicted = 0;
    }

    /// The number of cached entries.
    #[inline]
    pub fn len_cached(&self) -> usize {
        self.len_tracked() - self.len_evicted()
    }

    /// The number of evicted entries.
    #[inline]
    pub fn len_evicted(&self) -> usize {
        self.num_evicted
    }

    /// The total number of entries, cached or evicted.
    #[inline]
    pub fn len_tracked(&self) -> usize {
        self.store.len()
    }

    /// Iterate over the keys of all entries, cached or evicted.
    #[inline]
    pub fn keys(&self) -> LruCacheKeys<K, V, E> {
        self.store.keys()
    }

    /// Iterate over all `(key, entry)` pairs.
    #[inline]
    pub fn entries(&self) -> LruCacheEntries<K, V, E> {
        LruCacheEntries {
            inner: self.store.iter(),
        }
    }
}

impl<K, V, E, H> IntoIterator for LruCache<K, V, E, H>
where
    E: Copy,
{
    type IntoIter = LruCacheIntoIter<K, V, E>;
    type Item = (K, CacheEntry<V, E>);

    /// Consume `self` and iterate over all entries.
    #[inline]
    fn into_iter(self) -> LruCacheIntoIter<K, V, E> {
        LruCacheIntoIter {
            inner: self.store.into_iter(),
        }
    }
}

pub type LruCacheKeys<'a, K, V, E> = hash_map::Keys<'a, K, CacheEntry<(V, usize), E>>;
type LruCacheIter<'a, K, V, E> = hash_map::Iter<'a, K, CacheEntry<(V, usize), E>>;

pub struct LruCacheEntries<'a, K, V, E> {
    inner: LruCacheIter<'a, K, V, E>,
}

impl<'a, K, V, E> Iterator for LruCacheEntries<'a, K, V, E>
where
    E: Copy,
{
    type Item = (&'a K, CacheEntry<&'a V, E>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(k, e)| (k, e.as_ref().map_cached(|(v, _)| v)))
    }
}

pub struct LruCacheIntoIter<K, V, E> {
    inner: hash_map::IntoIter<K, CacheEntry<(V, usize), E>>,
}

impl<K, V, E> Iterator for LruCacheIntoIter<K, V, E>
where
    E: Copy,
{
    type Item = (K, CacheEntry<V, E>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(k, entry)| (k, entry.map_cached(|(val, _)| val)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheEntry<C, E> {
    Cached(C),
    Evicted(E),
}

impl<C, E> CacheEntry<C, E>
where
    E: Copy,
{
    #[inline]
    pub fn as_ref(&self) -> CacheEntry<&C, E> {
        match self {
            Self::Cached(c) => CacheEntry::Cached(c),
            Self::Evicted(e) => CacheEntry::Evicted(*e),
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> CacheEntry<&mut C, E> {
        match self {
            Self::Cached(c) => CacheEntry::Cached(c),
            Self::Evicted(e) => CacheEntry::Evicted(*e),
        }
    }

    #[inline]
    pub fn map_cached<T>(self, f: impl FnOnce(C) -> T) -> CacheEntry<T, E> {
        match self {
            Self::Cached(c) => CacheEntry::Cached(f(c)),
            Self::Evicted(e) => CacheEntry::Evicted(e),
        }
    }

    #[inline]
    pub fn some_if_cached(self) -> Option<C> {
        match self {
            Self::Cached(c) => Some(c),
            Self::Evicted(_) => None,
        }
    }

    #[inline]
    pub fn some_if_evicted(&self) -> Option<E> {
        match self {
            Self::Cached(_) => None,
            Self::Evicted(e) => Some(*e),
        }
    }

    #[inline]
    pub fn unwrap_evicted(&self) -> E {
        self.some_if_evicted().unwrap()
    }
}

impl<C, E> CacheEntry<(C, usize), E>
where
    E: Copy,
{
    #[inline]
    pub fn unwrap_value(self) -> C {
        self.some_if_cached().unwrap().0
    }
}

impl<'a, C, E> CacheEntry<&'a mut (C, usize), E>
where
    E: Copy,
{
    #[inline]
    pub fn unwrap_value(self) -> &'a mut C {
        &mut self.some_if_cached().unwrap().0
    }
}

/// Doubly-linked list using Vec as storage.
#[derive(Clone, Debug)]
struct LruList<T> {
    entries: Vec<ListEntry<T>>,
}

#[derive(Clone, Debug)]
struct ListEntry<T> {
    value: Option<T>,
    next: usize,
    prev: usize,
}

/// Free and occupied cells are each linked into a cyclic list with one auxiliary cell.
/// Cell #0 is on the list of free cells, element #1 is on the list of occupied cells.
impl<T> LruList<T> {
    const FREE: usize = 0;
    const OCCUPIED: usize = 1;

    fn new() -> LruList<T> {
        let mut entries = Vec::with_capacity(2);
        entries.push(ListEntry::<T> {
            value: None,
            next: 0,
            prev: 0,
        });
        entries.push(ListEntry::<T> {
            value: None,
            next: 1,
            prev: 1,
        });

        LruList { entries }
    }

    fn unlink(&mut self, index: usize) {
        let prev = self.entries[index].prev;
        let next = self.entries[index].next;
        self.entries[prev].next = next;
        self.entries[next].prev = prev;
    }

    fn link_after(&mut self, index: usize, prev: usize) {
        let next = self.entries[prev].next;
        self.entries[index].prev = prev;
        self.entries[index].next = next;
        self.entries[prev].next = index;
        self.entries[next].prev = index;
    }

    fn move_to_front(&mut self, index: usize) {
        self.unlink(index);
        self.link_after(index, Self::OCCUPIED);
    }

    fn push_front(&mut self, value: Option<T>) -> usize {
        if self.entries[Self::FREE].next == Self::FREE {
            self.entries.push(ListEntry::<T> {
                value: None,
                next: Self::FREE,
                prev: Self::FREE,
            });
            self.entries[Self::FREE].next = self.entries.len() - 1;
        }
        let index = self.entries[Self::FREE].next;
        self.entries[index].value = value;
        self.unlink(index);
        self.link_after(index, Self::OCCUPIED);

        index
    }

    fn remove(&mut self, index: usize) -> T {
        self.unlink(index);
        self.link_after(index, Self::FREE);

        self.entries[index].value.take().expect("invalid index")
    }

    fn back(&self) -> usize {
        self.entries[Self::OCCUPIED].prev
    }

    fn pop_back(&mut self) -> T {
        let index = self.back();

        self.remove(index)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.entries.push(ListEntry::<T> {
            value: None,
            next: 0,
            prev: 0,
        });
        self.entries.push(ListEntry::<T> {
            value: None,
            next: 1,
            prev: 1,
        });
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct EvictedLocation(usize);

    #[test]
    fn get_after_insert_and_evict_and_remove() {
        let mut cache = SmallKeyLruCache::default();
        assert_eq!(cache.get(&1), None);

        cache.insert(1, 2);
        assert_eq!(cache.get(&1), Some(CacheEntry::Cached(&2)));
        assert_eq!(cache.len_cached(), 1);

        cache.evict_lru(EvictedLocation(0));
        assert_eq!(cache.get(&1), Some(CacheEntry::Evicted(EvictedLocation(0))));
        assert_eq!(cache.len_evicted(), 1);

        cache.remove(&1);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.len_evicted(), 0);
    }

    #[test]
    fn get_after_insert_and_remove() {
        let mut cache: SmallKeyLruCache<i32, i32, ()> = SmallKeyLruCache::default();

        cache.insert(1, 2);

        cache.remove(&1);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.len_evicted(), 0);
    }

    #[test]
    fn repopulate_after_evict() {
        let mut cache = SmallKeyLruCache::default();
        assert_eq!(cache.get(&1), None);

        cache.insert(1, 2);
        cache.evict_lru(EvictedLocation(0));

        assert_eq!(
            cache.get_mut_or_repopulate_with(1, |loc| {
                assert_eq!(loc, EvictedLocation(0));
                3
            }),
            Some(&mut 3)
        );
        assert_eq!(cache.len_evicted(), 0);
    }

    #[test]
    fn evict_lru() {
        let mut cache = SmallKeyLruCache::default();

        cache.insert(1, 2);
        cache.insert(2, 3);
        cache.insert(3, 4);
        cache.insert(4, 5);
        cache.insert(2, 5);

        assert_eq!(cache.evict_lru(()), Some((1, 2)));
        assert_eq!(cache.evict_lru(()), Some((3, 4)));
        assert_eq!(cache.evict_lru(()), Some((4, 5)));
        assert_eq!(cache.evict_lru(()), Some((2, 5)));

        assert!(cache.len_cached() == 0);
        assert!(cache.len_evicted() == 4);
    }

    #[test]
    fn get_does_not_affect_lru_order() {
        let mut cache = SmallKeyLruCache::default();

        cache.insert(1, 2);
        cache.insert(2, 3);
        cache.get(&1);

        assert_eq!(cache.evict_lru(()), Some((1, 2)));
        assert_eq!(cache.evict_lru(()), Some((2, 3)));
    }

    #[test]
    fn evict_empty_entry_increases_num_evicted() {
        let mut cache = SmallKeyLruCache::default();

        cache.insert(1, 2);
        cache.remove(&1);
        cache.evict(1, ());

        assert_eq!(cache.len_evicted(), 1);
    }
}
