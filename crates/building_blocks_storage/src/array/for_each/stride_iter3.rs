use crate::{Iter3, Local3i, Stride};

use building_blocks_core::prelude::*;

pub(crate) struct Array3StrideIter {
    x_stride: usize,
    y_stride: usize,
    z_stride: usize,
    x_start: usize,
    y_start: usize,
    z_start: usize,
    x_i: usize,
    y_i: usize,
    z_i: usize,
}

impl Array3StrideIter {
    pub fn new_with_step(array_shape: Point3i, index_min: Local3i, step: Point3i) -> Self {
        debug_assert!(array_shape >= Point3i::ZERO);
        debug_assert!(index_min.0 >= Point3i::ZERO);
        debug_assert!(step >= Point3i::ZERO);

        let mut x_stride = 1usize;
        let mut y_stride = array_shape.x() as usize;
        let mut z_stride = (array_shape.y() * array_shape.x()) as usize;

        let x_start = x_stride * index_min.0.x() as usize;
        let y_start = y_stride * index_min.0.y() as usize;
        let z_start = z_stride * index_min.0.z() as usize;

        x_stride *= step.x() as usize;
        y_stride *= step.y() as usize;
        z_stride *= step.z() as usize;

        Self {
            x_stride,
            y_stride,
            z_stride,
            x_start,
            y_start,
            z_start,
            x_i: 0,
            y_i: 0,
            z_i: 0,
        }
    }

    pub fn new(array_shape: Point3i, index_min: Local3i) -> Self {
        Self::new_with_step(array_shape, index_min, Point3i::ONES)
    }
}

impl Iter3 for Array3StrideIter {
    type Coords = Stride;

    #[inline]
    fn coords(&self) -> Self::Coords {
        Stride(self.x_i)
    }

    #[inline]
    fn start_z(&mut self) {
        self.z_i = self.z_start;
    }
    #[inline]
    fn start_y(&mut self) {
        self.y_i = self.z_i + self.y_start;
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
    #[inline]
    fn incr_z(&mut self) {
        self.z_i += self.z_stride;
    }
}
