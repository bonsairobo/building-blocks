use crate::{ArrayIndexer, ArrayIterSpan, Local, Stride};

use building_blocks_core::prelude::*;

/// All information required to do strided iteration over an extent of a single array.
#[derive(Clone)]
pub struct ArrayForEach<N> {
    pub(crate) iter_extent: ExtentN<N>,
    pub(crate) span: ArrayIterSpan<N>,
}

/// A 2D `ArrayForEach`.
pub type Array2ForEach = ArrayForEach<[i32; 2]>;
/// A 3D `ArrayForEach`.
pub type Array3ForEach = ArrayForEach<[i32; 3]>;

impl<N> ArrayForEach<N>
where
    PointN<N>: IntegerPoint<N>,
{
    #[inline]
    pub fn new_local_unchecked(
        array_shape: PointN<N>,
        origin: Local<N>,
        iter_extent: ExtentN<N>,
    ) -> Self {
        Self {
            iter_extent,
            span: ArrayIterSpan {
                array_shape,
                origin,
                step: PointN::ONES,
            },
        }
    }

    #[inline]
    pub fn new_local(array_shape: PointN<N>, iter_extent: ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent =
            iter_extent.intersection(&ExtentN::from_min_and_shape(PointN::ZERO, array_shape));

        Self::new_local_unchecked(array_shape, Local(iter_extent.minimum), iter_extent)
    }

    #[inline]
    pub fn new_global_unchecked(array_extent: ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Translate to local coordinates.
        let origin = Local(iter_extent.minimum - array_extent.minimum);

        Self {
            iter_extent,
            span: ArrayIterSpan {
                array_shape: array_extent.shape,
                origin,
                step: PointN::ONES,
            },
        }
    }

    #[inline]
    pub fn new_global(array_extent: ExtentN<N>, iter_extent: ExtentN<N>) -> Self {
        // Make sure we don't index out of array bounds.
        let iter_extent = iter_extent.intersection(&array_extent);

        Self::new_global_unchecked(array_extent, iter_extent)
    }
}

impl<N> ArrayForEach<N>
where
    N: ArrayIndexer<N>,
{
    pub fn for_each(self, f: impl FnMut(PointN<N>, Stride)) {
        N::for_each(self, f)
    }
}
