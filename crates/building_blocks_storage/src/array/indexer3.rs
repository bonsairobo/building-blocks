use crate::array::{ArrayIndexer, Local, Local3i, Stride};

use building_blocks_core::prelude::*;

impl ArrayIndexer<[i32; 3]> for [i32; 3] {
    #[inline]
    fn stride_from_local_point(s: Point3i, p: Local3i) -> Stride {
        Stride((p.z() * s.y() * s.x() + p.y() * s.x() + p.x()) as usize)
    }

    #[inline]
    fn for_each_point_and_stride_unchecked(
        array_shape: Point3i,
        index_min: Local3i,
        iter_extent: Extent3i,
        mut f: impl FnMut(Point3i, Stride),
    ) {
        let iter_lub = iter_extent.least_upper_bound();
        let mut s = Array3ForEachState::new(array_shape, index_min);
        s.start_z();
        for z in iter_extent.minimum.z()..iter_lub.z() {
            s.start_y();
            for y in iter_extent.minimum.y()..iter_lub.y() {
                s.start_x();
                for x in iter_extent.minimum.x()..iter_lub.x() {
                    f(PointN([x, y, z]), s.stride());
                    s.incr_x();
                }
                s.incr_y();
            }
            s.incr_z();
        }
    }

    #[inline]
    fn for_each_stride_parallel_global_unchecked(
        iter_extent: &Extent3i,
        array1_extent: &Extent3i,
        array2_extent: &Extent3i,
        mut f: impl FnMut(Stride, Stride),
    ) {
        // Translate to local coordinates.
        let min1 = iter_extent.minimum - array1_extent.minimum;
        let min2 = iter_extent.minimum - array2_extent.minimum;

        let mut s1 = Array3ForEachState::new(array1_extent.shape, Local(min1));
        let mut s2 = Array3ForEachState::new(array2_extent.shape, Local(min2));

        s1.start_z();
        s2.start_z();
        for _z in 0..iter_extent.shape.z() {
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
            s1.incr_z();
            s2.incr_z();
        }
    }
}

struct Array3ForEachState {
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

impl Array3ForEachState {
    fn new(array_shape: Point3i, index_min: Local3i) -> Self {
        let x_stride = 1usize;
        let y_stride = array_shape.x() as usize;
        let z_stride = (array_shape.y() * array_shape.x()) as usize;
        let x_start = x_stride * index_min.0.x() as usize;
        let y_start = y_stride * index_min.0.y() as usize;
        let z_start = z_stride * index_min.0.z() as usize;

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

    fn stride(&self) -> Stride {
        Stride(self.x_i)
    }

    fn start_z(&mut self) {
        self.z_i = self.z_start;
    }
    fn start_y(&mut self) {
        self.y_i = self.z_i + self.y_start;
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
    fn incr_z(&mut self) {
        self.z_i += self.z_stride;
    }
}
