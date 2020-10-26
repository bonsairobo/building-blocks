use crate::{Array, ArrayN, Local, Stride};

use building_blocks_core::prelude::*;

/// Map-local coordinates, wrapping a `Point2i`.
pub type Local2i = Local<[i32; 2]>;

pub type Array2<T> = ArrayN<[i32; 2], T>;

impl<T> Array<[i32; 2]> for Array2<T> {
    #[inline]
    fn stride_from_point_static(s: &Point2i, p: &Point2i) -> Stride {
        Stride((p.y() * s.x() + p.x()) as usize)
    }

    fn for_each_point_and_stride(
        array_extent: &Extent2i,
        extent: &Extent2i,
        mut f: impl FnMut(Point2i, Stride),
    ) {
        // Translate to local coordinates.
        let global_extent = extent.intersection(array_extent);
        let global_lub = global_extent.least_upper_bound();
        let local_extent = global_extent - array_extent.minimum;

        let mut s = Array2ForEachState::new(&array_extent.shape, &Local(local_extent.minimum));
        s.start_y();
        for y in global_extent.minimum.y()..global_lub.y() {
            s.start_x();
            for x in global_extent.minimum.x()..global_lub.x() {
                f(PointN([x, y]), s.stride());
                s.incr_x();
            }
            s.incr_y();
        }
    }

    fn for_each_stride_parallel(
        iter_extent: &Extent2i,
        array1_extent: &Extent2i,
        array2_extent: &Extent2i,
        mut f: impl FnMut(Stride, Stride),
    ) {
        // Translate to local coordinates.
        let min1 = iter_extent.minimum - array1_extent.minimum;
        let min2 = iter_extent.minimum - array2_extent.minimum;

        let mut s1 = Array2ForEachState::new(&array1_extent.shape, &Local(min1));
        let mut s2 = Array2ForEachState::new(&array2_extent.shape, &Local(min2));

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
    fn new(array_shape: &Point2i, iter_min: &Local2i) -> Self {
        let x_stride = 1usize;
        let y_stride = array_shape.x() as usize;
        let x_start = x_stride * iter_min.0.x() as usize;
        let y_start = y_stride * iter_min.0.y() as usize;

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
