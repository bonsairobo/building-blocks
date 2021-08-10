mod chunk_db;
mod delta_batch;
mod key;
mod read_result;

#[cfg(feature = "sled-snapshots")]
mod versioned_chunk_db;

pub use chunk_db::*;
pub use delta_batch::*;
pub use key::*;
pub use read_result::*;

#[cfg(feature = "sled-snapshots")]
pub use versioned_chunk_db::*;

pub use sled;

#[cfg(feature = "sled-snapshots")]
pub use sled_snapshots;

pub enum Delta<K, V> {
    Insert(K, V),
    Remove(K),
}

impl<K, V> Delta<K, V> {
    fn key(&self) -> &K {
        match self {
            Self::Insert(k, _) => &k,
            Self::Remove(k) => &k,
        }
    }
}
