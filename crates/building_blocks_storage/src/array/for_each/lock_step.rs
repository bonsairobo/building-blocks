use crate::array::{ArrayIndexer, ArrayStrideIter, Local};

use building_blocks_core::prelude::*;

/// All information required to do strided iteration over two arrays in lock step.
///
/// This means that the same extent will be iterated for both arrays, but each array may interpret that extent differently. For
/// example, one array might have a different step size or local origin than the other, causing it to visit different points
/// than the actual `iter_extent`. For this reason, `iter_extent` should only be used as a shared reference point.
#[derive(Clone)]
pub struct LockStepArrayForEach<N> {
    pub(crate) iter_extent: ExtentN<N>,
    pub(crate) iter1: ArrayStrideIter,
    pub(crate) iter2: ArrayStrideIter,
}

pub type LockStepArrayForEach2 = LockStepArrayForEach<[i32; 2]>;
pub type LockStepArrayForEach3 = LockStepArrayForEach<[i32; 3]>;

impl<N> LockStepArrayForEach<N>
where
    N: ArrayIndexer<N>,
    PointN<N>: IntegerPoint,
{
    pub fn new(iter_extent: ExtentN<N>, iter1: ArrayStrideIter, iter2: ArrayStrideIter) -> Self {
        Self {
            iter_extent,
            iter1,
            iter2,
        }
    }

    pub fn new_global_unchecked(
        iter_extent: ExtentN<N>,
        array1_extent: ExtentN<N>,
        array2_extent: ExtentN<N>,
    ) -> Self {
        // Translate to local coordinates.
        let origin1 = iter_extent.minimum - array1_extent.minimum;
        let origin2 = iter_extent.minimum - array2_extent.minimum;

        let iter1 = N::make_stride_iter(array1_extent.shape, Local(origin1), PointN::ONES);
        let iter2 = N::make_stride_iter(array2_extent.shape, Local(origin2), PointN::ONES);

        Self::new(iter_extent, iter1, iter2)
    }
}
