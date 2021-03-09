#[macro_use]
mod for_each2;
#[macro_use]
mod for_each3;

pub(crate) use for_each2::{for_each_stride_parallel_global_unchecked2, Array2x1ForEachState};
pub(crate) use for_each3::{for_each_stride_parallel_global_unchecked3, Array3x1ForEachState};

use crate::{ArrayIndexer, Local, Stride};

use building_blocks_core::prelude::*;

/// All information required to do strided iteration over an extent.
#[derive(Clone)]
pub struct ArrayForEach<N> {
    /// Extent of the iteration coordinates.
    pub(crate) iter_extent: ExtentN<N>,
    /// Shape of the array being indexed.
    pub(crate) array_shape: PointN<N>,
    /// Array-local minimum where we start indexing.
    pub(crate) index_min: Local<N>,
}

/// A 2D `ArrayForEach`.
pub type Array2x1ForEach = ArrayForEach<[i32; 2]>;
/// A 3D `ArrayForEach`.
pub type Array3x1ForEach = ArrayForEach<[i32; 3]>;

impl<N> ArrayForEach<N>
where
    PointN<N>: IntegerPoint<N>,
{
    #[inline]
    pub fn new_local_unchecked(
        array_shape: PointN<N>,
        index_min: Local<N>,
        iter_shape: PointN<N>,
    ) -> Self {
        Self {
            iter_extent: ExtentN::from_min_and_shape(index_min.0, iter_shape),
            array_shape,
            index_min,
        }
    }

    #[inline]
    pub fn new_local(array_shape: PointN<N>, iter_extent: &ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent =
            iter_extent.intersection(&ExtentN::from_min_and_shape(PointN::ZERO, array_shape));

        Self::new_local_unchecked(array_shape, Local(iter_extent.minimum), iter_extent.shape)
    }

    #[inline]
    pub fn new_global_unchecked(array_extent: &ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Translate to local coordinates.
        let index_min = Local(iter_extent.minimum - array_extent.minimum);

        Self {
            iter_extent,
            array_shape: array_extent.shape,
            index_min,
        }
    }

    #[inline]
    pub fn new_global(array_extent: &ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(array_extent);

        Self::new_global_unchecked(array_extent, iter_extent)
    }
}

impl<N> ArrayForEach<N>
where
    N: ArrayIndexer<N>,
    PointN<N>: Copy,
{
    pub fn for_each_point_and_stride(self, f: impl FnMut(PointN<N>, Stride)) {
        N::for_each_point_and_stride_unchecked(self, f)
    }
}
