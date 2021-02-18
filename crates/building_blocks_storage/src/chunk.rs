mod compression;
mod serialization;

pub use compression::*;
pub use serialization::*;

use crate::array::ArrayN;

use serde::{Deserialize, Serialize};

/// One piece of a chunked lattice map. Contains both some generic metadata and the data for each point in the chunk extent.
#[derive(Clone, Deserialize, Serialize)]
pub struct Chunk<N, T, Meta = ()> {
    pub metadata: Meta,
    pub array: ArrayN<N, T>,
}

/// A 2-dimensional `Chunk`.
pub type Chunk2<T, Meta = ()> = Chunk<[i32; 2], T, Meta>;
/// A 3-dimensional `Chunk`.
pub type Chunk3<T, Meta = ()> = Chunk<[i32; 3], T, Meta>;

impl<N, T> Chunk<N, T, ()> {
    /// Constructs a chunk without metadata.
    pub fn with_array(array: ArrayN<N, T>) -> Self {
        Chunk {
            metadata: (),
            array,
        }
    }
}
