use crate::{Array2ForEach, Local, Local2i, Stride};

use building_blocks_core::prelude::*;

pub trait StrideIter2 {
    type Stride;

    fn stride(&self) -> Self::Stride;

    fn start_y(&mut self);
    fn start_x(&mut self);

    fn incr_x(&mut self);
    fn incr_y(&mut self);
}

impl Array2ForEach {
    #[inline]
    pub fn for_each_point_and_stride_unchecked(self, f: impl FnMut(Point2i, Stride)) {
        let Array2ForEach {
            iter_extent,
            array_shape,
            index_min,
        } = self;
        for_each2(
            Array2ForEachState::new(array_shape, index_min),
            &iter_extent,
            f,
        );
    }
}

#[inline]
pub fn for_each2<S>(mut s: S, iter_extent: &Extent2i, mut f: impl FnMut(Point2i, S::Stride))
where
    S: StrideIter2,
{
    let iter_lub = iter_extent.least_upper_bound();
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
pub(crate) fn for_each_stride_lockstep_global_unchecked2(
    iter_extent: &Extent2i,
    array1_extent: &Extent2i,
    array2_extent: &Extent2i,
    mut f: impl FnMut(Stride, Stride),
) {
    // Translate to local coordinates.
    let min1 = iter_extent.minimum - array1_extent.minimum;
    let min2 = iter_extent.minimum - array2_extent.minimum;

    let s1 = Array2ForEachState::new(array1_extent.shape, Local(min1));
    let s2 = Array2ForEachState::new(array2_extent.shape, Local(min2));

    for_each2((s1, s2), iter_extent, |_p, (s1, s2)| f(s1, s2));
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
}

impl StrideIter2 for Array2ForEachState {
    type Stride = Stride;

    #[inline]
    fn stride(&self) -> Self::Stride {
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

macro_rules! impl_stride_iter2_for_tuple {
    ( $( $var:ident : $t:ident ),+ ) => {
        impl<$($t),+> StrideIter2 for ($($t,)+)
        where
            $($t: StrideIter2),+
        {
            type Stride = ($($t::Stride,)+);

            #[inline]
            fn stride(&self) -> Self::Stride {
                let ($($var,)+) = self;

                ($($var.stride(),)+)
            }

            #[inline]
            fn start_y(&mut self) {
                let ($($var,)+) = self;
                $( $var.start_y(); )+
            }
            #[inline]
            fn start_x(&mut self) {
                let ($($var,)+) = self;
                $( $var.start_x(); )+
            }

            #[inline]
            fn incr_x(&mut self) {
                let ($($var,)+) = self;
                $( $var.incr_x(); )+
            }
            #[inline]
            fn incr_y(&mut self) {
                let ($($var,)+) = self;
                $( $var.incr_y(); )+
            }
        }
    };
}

impl_stride_iter2_for_tuple! { a: A }
impl_stride_iter2_for_tuple! { a: A, b: B }
impl_stride_iter2_for_tuple! { a: A, b: B, c: C }
impl_stride_iter2_for_tuple! { a: A, b: B, c: C, d: D }
impl_stride_iter2_for_tuple! { a: A, b: B, c: C, d: D, e: E }
impl_stride_iter2_for_tuple! { a: A, b: B, c: C, d: D, e: E, f: F }
