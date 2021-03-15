use crate::{AsMutRef, Get, GetMut, GetMutPtr, GetRef};

macro_rules! impl_for_tuple {
    ( $( $var:ident : $t:ident ),+ ) => {
        impl<Coord, $($t),+> Get<Coord> for ($($t,)+)
        where
            Coord: Copy,
            $($t: Get<Coord>),+
        {
            type Item = ($($t::Item,)+);

            #[inline]
            fn get(&self, offset: Coord) -> Self::Item {
                let ($($var,)+) = self;

                ($($var.get(offset),)+)
            }
        }

        impl<'a, Coord, $($t),+> GetRef<'a, Coord> for ($($t,)+)
        where
            Coord: Copy,
            $($t: GetRef<'a, Coord>),+
        {
            type Item = ($($t::Item,)+);

            #[inline]
            fn get_ref(&'a self, offset: Coord) -> Self::Item {
                let ($($var,)+) = self;

                ($($var.get_ref(offset),)+)
            }
        }

        impl<'a, Coord, $($t),+> GetMut<'a, Coord> for ($($t,)+)
        where
            Coord: Copy,
            $($t: GetMut<'a, Coord>),+
        {
            type Item = ($($t::Item,)+);

            #[inline]
            fn get_mut(&'a mut self, offset: Coord) -> Self::Item {
                let ($($var,)+) = self;

                ($($var.get_mut(offset),)+)
            }
        }

        impl<Coord, $($t),+> GetMutPtr<Coord> for ($($t,)+)
        where
            Coord: Copy,
            $($t: GetMutPtr<Coord>),+
        {
            type Item = ($($t::Item,)+);

            #[inline]
            unsafe fn get_mut_ptr(&mut self, offset: Coord) -> Self::Item {
                let ($($var,)+) = self;

                ($($var.get_mut_ptr(offset),)+)
            }
        }

        impl<'a, $($t),+> AsMutRef<'a> for ($(*mut $t,)+)
        where
            $($t: 'a,)+
        {
            type MutRef = ($(&'a mut $t,)+);

            #[inline]
            fn as_mut_ref(self) -> Self::MutRef {
                let ($($var,)+) = self;

                unsafe { ($(&mut *$var,)+) }
            }
        }
    };
}

impl_for_tuple! { a: A }
impl_for_tuple! { a: A, b: B }
impl_for_tuple! { a: A, b: B, c: C }
impl_for_tuple! { a: A, b: B, c: C, d: D }
impl_for_tuple! { a: A, b: B, c: C, d: D, e: E }
impl_for_tuple! { a: A, b: B, c: C, d: D, e: E, f: F }
