#[macro_use]
mod for_each2;
#[macro_use]
mod for_each3;

mod single_array;
mod stride_iter2;
mod stride_iter3;

pub use single_array::*;

pub(crate) use for_each2::*;
pub(crate) use for_each3::*;
pub(crate) use stride_iter2::*;
pub(crate) use stride_iter3::*;

use crate::Local;
use building_blocks_core::PointN;

#[derive(Clone)]
pub(crate) struct ForEachSpan<N> {
    /// Array-local point where we start iteration.
    pub origin: Local<N>,
    /// The step size taken in each dimension.
    pub step: PointN<N>,
}
