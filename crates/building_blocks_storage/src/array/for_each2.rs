use crate::{Local, Local2i, Stride};

use building_blocks_core::prelude::*;

/// Execute a block of code in a strided 2D for loop described by `for_each`.
macro_rules! for_each2 {
    ($for_each:ident, $x:ident, $y:ident, $stride:ident, $code:block) => {
        let $crate::Array2ForEach {
            iter_extent,
            array_shape,
            index_min,
        } = $for_each;
        let iter_lub = iter_extent.least_upper_bound();
        let mut s = $crate::Array2ForEachState::new(array_shape, index_min);
        s.start_y();
        for $y in iter_extent.minimum.y()..iter_lub.y() {
            s.start_x();
            for $x in iter_extent.minimum.x()..iter_lub.x() {
                let $stride = s.stride();
                $code;
                s.incr_x();
            }
            s.incr_y();
        }
    };
}

#[inline]
pub(crate) fn for_each_stride_parallel_global_unchecked2(
    iter_extent: &Extent2i,
    array1_extent: &Extent2i,
    array2_extent: &Extent2i,
    mut f: impl FnMut(Stride, Stride),
) {
    // Translate to local coordinates.
    let min1 = iter_extent.minimum - array1_extent.minimum;
    let min2 = iter_extent.minimum - array2_extent.minimum;

    let mut s1 = Array2ForEachState::new(array1_extent.shape, Local(min1));
    let mut s2 = Array2ForEachState::new(array2_extent.shape, Local(min2));

    s1.start_y();
    s2.start_y();
    for _y in 0..iter_extent.shape.y() {
        s1.start_x();
        s2.start_x();
        for _x in 0..iter_extent.shape.x() {
            f(s1.stride(), s2.stride());

            s1.incr_x();
            s2.incr_x();
        }
        s1.incr_y();
        s2.incr_y();
    }
}

pub(crate) struct Array2ForEachState {
    x_stride: usize,
    y_stride: usize,
    x_start: usize,
    y_start: usize,
    x_i: usize,
    y_i: usize,
}

impl Array2ForEachState {
    pub(crate) fn new(array_shape: Point2i, index_min: Local2i) -> Self {
        let x_stride = 1usize;
        let y_stride = array_shape.x() as usize;
        let x_start = x_stride * index_min.0.x() as usize;
        let y_start = y_stride * index_min.0.y() as usize;

        Self {
            x_stride,
            y_stride,
            x_start,
            y_start,
            x_i: 0,
            y_i: 0,
        }
    }

    pub(crate) fn stride(&self) -> Stride {
        Stride(self.x_i)
    }

    pub(crate) fn start_y(&mut self) {
        self.y_i = self.y_start;
    }
    pub(crate) fn start_x(&mut self) {
        self.x_i = self.y_i + self.x_start;
    }

    pub(crate) fn incr_x(&mut self) {
        self.x_i += self.x_stride;
    }
    pub(crate) fn incr_y(&mut self) {
        self.y_i += self.y_stride;
    }
}
