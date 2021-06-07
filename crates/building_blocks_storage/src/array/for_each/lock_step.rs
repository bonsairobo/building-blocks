use crate::{ArrayIterSpan, Local};

use building_blocks_core::prelude::*;

/// All information required to do strided iteration over two arrays in lock step.
///
/// This means that the same extent will be iterated for both arrays, but each array may interpret that extent differently. For
/// example, one array might have a different step size or local origin than the other, causing it to visit different points
/// than the actual `iter_extent`. For this reason, `iter_extent` should only be used as a shared reference point.
#[derive(Clone)]
pub struct LockStepArrayForEach<N> {
    pub(crate) iter_extent: ExtentN<N>,
    pub(crate) span1: ArrayIterSpan<N>,
    pub(crate) span2: ArrayIterSpan<N>,
}

pub type LockStepArrayForEach2 = LockStepArrayForEach<[i32; 2]>;
pub type LockStepArrayForEach3 = LockStepArrayForEach<[i32; 3]>;

impl<N> LockStepArrayForEach<N>
where
    PointN<N>: IntegerPoint<N>,
{
    pub fn new_global_unchecked(
        iter_extent: ExtentN<N>,
        array1_extent: ExtentN<N>,
        array2_extent: ExtentN<N>,
    ) -> Self {
        // Translate to local coordinates.
        let origin1 = iter_extent.minimum - array1_extent.minimum;
        let origin2 = iter_extent.minimum - array2_extent.minimum;

        LockStepArrayForEach {
            iter_extent,
            span1: ArrayIterSpan {
                array_shape: array1_extent.shape,
                origin: Local(origin1),
                step: PointN::ONES,
            },
            span2: ArrayIterSpan {
                array_shape: array2_extent.shape,
                origin: Local(origin2),
                step: PointN::ONES,
            },
        }
    }
}
