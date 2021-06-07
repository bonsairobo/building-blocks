use crate::{Iter2, Local2i, Stride};

use building_blocks_core::prelude::*;

pub(crate) struct Array2StrideIter {
    x_stride: usize,
    y_stride: usize,
    x_start: usize,
    y_start: usize,
    x_i: usize,
    y_i: usize,
}

impl Array2StrideIter {
    pub fn new_with_step(array_shape: Point2i, origin: Local2i, step: Point2i) -> Self {
        debug_assert!(array_shape >= Point2i::ZERO);
        debug_assert!(origin.0 >= Point2i::ZERO);
        debug_assert!(step >= Point2i::ZERO);

        let mut x_stride = 1usize;
        let mut y_stride = array_shape.x() as usize;

        let x_start = x_stride * origin.0.x() as usize;
        let y_start = y_stride * origin.0.y() as usize;

        x_stride *= step.x() as usize;
        y_stride *= step.y() as usize;

        Self {
            x_stride,
            y_stride,
            x_start,
            y_start,
            x_i: 0,
            y_i: 0,
        }
    }

    pub fn new(array_shape: Point2i, origin: Local2i) -> Self {
        Self::new_with_step(array_shape, origin, Point2i::ONES)
    }
}

impl Iter2 for Array2StrideIter {
    type Coords = Stride;

    #[inline]
    fn coords(&self) -> Self::Coords {
        Stride(self.x_i)
    }

    #[inline]
    fn start_y(&mut self) {
        self.y_i = self.y_start;
    }
    #[inline]
    fn start_x(&mut self) {
        self.x_i = self.y_i + self.x_start;
    }

    #[inline]
    fn incr_x(&mut self) {
        self.x_i += self.x_stride;
    }
    #[inline]
    fn incr_y(&mut self) {
        self.y_i += self.y_stride;
    }
}
