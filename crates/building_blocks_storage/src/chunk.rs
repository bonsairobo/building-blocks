pub mod indexer;
pub mod map;
pub mod storage;

pub use indexer::*;
pub use map::*;
pub use storage::*;

use building_blocks_core::{IntegerPoint, PointN};
use serde::{Deserialize, Serialize};

/// A newtype wrapper for `PointN` or `ExtentN` where each point represents exactly one chunk.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ChunkUnits<T>(pub T);

impl<N> ChunkUnits<PointN<N>>
where
    PointN<N>: IntegerPoint<N>,
{
    pub fn chunk_min(&self, chunk_shape: PointN<N>) -> PointN<N> {
        chunk_shape * self.0
    }
}
