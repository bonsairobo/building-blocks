use core::mem::MaybeUninit;

/// Used for variadic conversion from `&(A, B, ...)` to `(&A, &B, ...)`.
pub trait MultiRef<'a> {
    type Data;

    fn from_data_ref(data_ref: &'a Self::Data) -> Self;
}

impl<'a, T> MultiRef<'a> for &'a T {
    type Data = T;

    #[inline]
    fn from_data_ref(data_ref: &'a Self::Data) -> Self {
        data_ref
    }
}

/// Used for variadic conversion from `(*mut A, *mut B, ...)` to `(&'a mut A, &'a mut B, ...)`.
pub trait IntoMultiMut<'a> {
    type MultiMut;

    fn into_multi_mut(self) -> Self::MultiMut;
}

impl<'a, T> IntoMultiMut<'a> for *mut T
where
    T: 'a,
{
    type MultiMut = &'a mut T;

    #[inline]
    fn into_multi_mut(self) -> Self::MultiMut {
        unsafe { &mut *self }
    }
}

/// Used for variadic conversion from `(*mut MaybeUninit<A>, *mut MaybeUninit<B>, ...)` to `(*mut A, *mut B)`.
pub trait IntoMultiMutPtr {
    type Data;
    type Ptr: MultiMutPtr<Data = Self::Data>;

    /// # Safety
    /// `Self::Ptr` is intended to contain `*mut` pointers, so this carries the same safety concerns as any such pointer.
    unsafe fn into_multi_mut_ptr(self) -> Self::Ptr;
}

impl<T> IntoMultiMutPtr for *mut MaybeUninit<T> {
    type Data = T;
    type Ptr = *mut T;

    #[inline]
    unsafe fn into_multi_mut_ptr(self) -> Self::Ptr {
        (&mut *self).as_mut_ptr()
    }
}

/// Used for variadic copying of source data `(A, B, ...)` to destination pointers `(*mut A, *mut B, ...)`.
pub trait MultiMutPtr {
    type Data;

    /// # Safety
    /// `self` is intended to contain `*mut` pointers, so this carries the same safety concerns as any such pointer.
    unsafe fn write(self, data: Self::Data);
}

impl<T> MultiMutPtr for *mut T {
    type Data = T;

    #[inline]
    unsafe fn write(self, data: Self::Data) {
        self.write(data);
    }
}

macro_rules! impl_tuple {
    ( $( $var1:ident, $var2:ident : $t:ident ),+ ) => {
        impl<'a, $($t),+> MultiRef<'a> for ($($t,)+)
        where
            $($t: 'a + MultiRef<'a>,)+
        {
            type Data = ($($t::Data,)+);

            #[inline]
            fn from_data_ref(data_ref: &'a Self::Data) -> Self {
                let ($($var1,)+) = data_ref;

                ($($t::from_data_ref($var1),)+)
            }
        }

        impl<'a, $($t),+> IntoMultiMut<'a> for ($(*mut $t,)+)
        where
            $($t: 'a,)+
        {
            type MultiMut = ($(&'a mut $t,)+);

            #[inline]
            fn into_multi_mut(self) -> Self::MultiMut {
                let ($($var1,)+) = self;

                unsafe { ($(&mut *$var1,)+) }
            }
        }

        impl<$($t),+> IntoMultiMutPtr for ($($t,)+)
        where
            $($t: IntoMultiMutPtr),+
        {
            type Data = ($($t::Data,)+);
            type Ptr = ($($t::Ptr,)+);

            #[inline]
            unsafe fn into_multi_mut_ptr(self) -> Self::Ptr {
                let ($($var1,)+) = self;

                ($($var1.into_multi_mut_ptr(),)+)
            }
        }

        impl<$($t),+> MultiMutPtr for ($($t,)+)
        where
            $($t: MultiMutPtr,)+
        {
            type Data = ($($t::Data,)+);

            #[inline]
            unsafe fn write(self, data: Self::Data) {
                let ($($var1,)+) = self;
                let ($($var2,)+) = data;

                $( $var1.write($var2); )+
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
