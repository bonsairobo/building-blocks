use core::mem::MaybeUninit;

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

/// For converting various pointer types into something writeable by multichannel arrays.
#[doc(hidden)]
pub trait AsWritePtr {
    type Data;
    type Ptr: WritePtr<Data = Self::Data>;

    unsafe fn as_mut_ptr(self) -> Self::Ptr;
}

impl<T> AsWritePtr for *mut MaybeUninit<T> {
    type Data = T;
    type Ptr = *mut T;

    #[inline]
    unsafe fn as_mut_ptr(self) -> Self::Ptr {
        MaybeUninit::as_mut_ptr(&mut *self)
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

        impl<$($t),+> AsWritePtr for ($($t,)+)
        where
            $($t: AsWritePtr),+
        {
            type Data = ($($t::Data,)+);
            type Ptr = ($($t::Ptr,)+);

            #[inline]
            unsafe fn as_mut_ptr(self) -> Self::Ptr {
                let ($($var1,)+) = self;

                ($($var1.as_mut_ptr(),)+)
            }
        }

        impl<$($t),+> WritePtr for ($($t,)+)
        where
            $($t: WritePtr,)+
        {
            type Data = ($($t::Data,)+);

            #[inline]
            unsafe fn write_ptr(self, data: Self::Data) {
                let ($($var1,)+) = self;
                let ($($var2,)+) = data;

                $( $var1.write_ptr($var2); )+
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
