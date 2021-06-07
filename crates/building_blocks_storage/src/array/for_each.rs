#[macro_use]
mod for_each2;
#[macro_use]
mod for_each3;

mod lock_step;
mod single_array;
mod stride_iter2;
mod stride_iter3;

pub use lock_step::*;
pub use single_array::*;

pub(crate) use for_each2::*;
pub(crate) use for_each3::*;
pub(crate) use stride_iter2::*;
pub(crate) use stride_iter3::*;

use crate::Local;
use building_blocks_core::PointN;

/// Information that may be specific to a single array during iteration.
#[derive(Clone)]
pub(crate) struct ArrayIterSpan<N> {
    /// Shape of the array being indexed.
    pub array_shape: PointN<N>,
    /// Array-local point where we start iteration.
    pub origin: Local<N>,
    /// The step size taken in each dimension.
    pub step: PointN<N>,
}
