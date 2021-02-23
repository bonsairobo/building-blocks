use crate::array::{ArrayIndexer, Local, Local2i, Stride};

use building_blocks_core::prelude::*;

impl ArrayIndexer<[i32; 2]> for [i32; 2] {
    #[inline]
    fn stride_from_local_point(s: Point2i, p: Local2i) -> Stride {
        Stride((p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each_point_and_stride_unchecked(
        array_shape: Point2i,
        index_min: Local2i,
        iter_extent: Extent2i,
        mut f: impl FnMut(Point2i, Stride),
    ) {
        let iter_lub = iter_extent.least_upper_bound();
        let mut s = Array2ForEachState::new(array_shape, index_min);
        s.start_y();
        for y in iter_extent.minimum.y()..iter_lub.y() {
            s.start_x();
            for x in iter_extent.minimum.x()..iter_lub.x() {
                f(PointN([x, y]), s.stride());
                s.incr_x();
            }
            s.incr_y();
        }
    }

    #[inline]
    fn for_each_stride_parallel_global_unchecked(
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
}

struct Array2ForEachState {
    x_stride: usize,
    y_stride: usize,
    x_start: usize,
    y_start: usize,
    x_i: usize,
    y_i: usize,
}

impl Array2ForEachState {
    fn new(array_shape: Point2i, index_min: Local2i) -> Self {
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

    fn stride(&self) -> Stride {
        Stride(self.x_i)
    }

    fn start_y(&mut self) {
        self.y_i = self.y_start;
    }
    fn start_x(&mut self) {
        self.x_i = self.y_i + self.x_start;
    }

    fn incr_x(&mut self) {
        self.x_i += self.x_stride;
    }
    fn incr_y(&mut self) {
        self.y_i += self.y_stride;
    }
}