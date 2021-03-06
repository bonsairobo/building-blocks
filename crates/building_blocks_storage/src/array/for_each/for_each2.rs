use building_blocks_core::prelude::*;

/// Steps a generic 2D iterator `iter` through some "extent," for some interpretation of an extent determined by the iterator.
///
/// The visitor `f` will see every point in `extent`, as well as whatever coordinates `iter` would like to associated with
/// that point.
#[inline]
pub fn for_each2<I>(mut iter: I, extent: &Extent2i, mut f: impl FnMut(Point2i, I::Coords))
where
    I: Iter2,
{
    let min = extent.minimum;
    let lub = extent.least_upper_bound();
    iter.start_y();
    for y in min.y()..lub.y() {
        iter.start_x();
        for x in min.x()..lub.x() {
            f(PointN([x, y]), iter.coords());
            iter.incr_x();
        }
        iter.incr_y();
    }
}

pub trait Iter2 {
    type Coords;

    fn coords(&self) -> Self::Coords;

    fn start_y(&mut self);
    fn start_x(&mut self);

    fn incr_x(&mut self);
    fn incr_y(&mut self);
}

macro_rules! impl_iter2_for_tuple {
    ( $( $var:ident : $t:ident ),+ ) => {
        impl<$($t),+> Iter2 for ($($t,)+)
        where
            $($t: Iter2),+
        {
            type Coords = ($($t::Coords,)+);

            #[inline]
            fn coords(&self) -> Self::Coords {
                let ($($var,)+) = self;

                ($($var.coords(),)+)
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

impl_iter2_for_tuple! { a: A }
impl_iter2_for_tuple! { a: A, b: B }
impl_iter2_for_tuple! { a: A, b: B, c: C }
impl_iter2_for_tuple! { a: A, b: B, c: C, d: D }
impl_iter2_for_tuple! { a: A, b: B, c: C, d: D, e: E }
impl_iter2_for_tuple! { a: A, b: B, c: C, d: D, e: E, f: F }
