/// An implementation detail of multichannel accessors. Sometimes we need to be able to transmute lifetimes to satisfy the
/// borrow checker, so we just get raw pointers and then convert them to borrows.
#[doc(hidden)]
pub trait AsMutRef<'a> {
    type MutRef;

    fn as_mut_ref(self) -> Self::MutRef;
}

impl<'a, T> AsMutRef<'a> for *mut T
where
    T: 'a,
{
    type MutRef = &'a mut T;

    #[inline]
    fn as_mut_ref(self) -> Self::MutRef {
        unsafe { &mut *self }
    }
}

/// An abstraction over multichannel pointers, i.e. tuples of pointers.
#[doc(hidden)]
pub trait WritePtr {
    type Data;

    unsafe fn write_ptr(self, data: Self::Data);
}

impl<T> WritePtr for *mut T {
    type Data = T;

    #[inline]
    unsafe fn write_ptr(self, data: Self::Data) {
        *self = data;
    }
}

macro_rules! impl_tuple {
    ( $( $var1:ident, $var2:ident : $t:ident ),+ ) => {
        impl<'a, $($t),+> AsMutRef<'a> for ($(*mut $t,)+)
        where
            $($t: 'a,)+
        {
            type MutRef = ($(&'a mut $t,)+);

            #[inline]
            fn as_mut_ref(self) -> Self::MutRef {
                let ($($var1,)+) = self;

                unsafe { ($(&mut *$var1,)+) }
            }
        }

        impl<$($t),+> WritePtr for ($(*mut $t,)+) {
            type Data = ($($t,)+);

            #[inline]
            unsafe fn write_ptr(self, data: Self::Data) {
                let ($($var1,)+) = self;
                let ($($var2,)+) = data;

                $( *$var1 = $var2; )+
            }
        }
    };
}

impl_tuple! { a1, a2: A }
impl_tuple! { a1, a2: A, b1, b2: B }
impl_tuple! { a1, a2: A, b1, b2: B, c1, c2: C }
impl_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D }
impl_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D, e1, e2: E }
impl_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D, e1, e2: E, f1, f2: F }
