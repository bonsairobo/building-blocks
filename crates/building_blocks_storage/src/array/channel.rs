use crate::{
    AsRawBytes, BytesCompression, Compressed, Compression, FromBytesCompression, GetMut, GetMutPtr,
    GetRef, MultiMutPtr,
};

use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Channel<T, Store = Vec<T>> {
    store: Store,
    marker: std::marker::PhantomData<T>,
}

impl<T, Store> Channel<T, Store> {
    #[inline]
    pub fn new(store: Store) -> Self {
        Self {
            store,
            marker: Default::default(),
        }
    }

    #[inline]
    pub fn take_store(self) -> Store {
        self.store
    }

    #[inline]
    pub fn store(&self) -> &Store {
        &self.store
    }

    #[inline]
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }
}

impl<T> Channel<T, Vec<T>> {
    pub fn fill(value: T, length: usize) -> Self
    where
        T: Clone,
    {
        Self::new(vec![value; length])
    }
}

impl<T, Store> Channel<T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    #[inline]
    pub fn reset_values(&mut self, value: T)
    where
        T: Clone,
    {
        self.store.fill(value);
    }
}

impl<'a, T, Store> AsRawBytes<'a> for Channel<T, Store>
where
    T: 'static + Copy,
    Store: Deref<Target = [T]>,
{
    type Output = &'a [u8];

    fn as_raw_bytes(&'a self) -> Self::Output {
        self.store().as_raw_bytes()
    }
}

//  ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
// ██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
// ██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
// ██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
// ╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
//  ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝

impl<'a, T, Store> GetRef<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: Deref<Target = [T]>,
{
    type Item = &'a T;

    #[inline]
    fn get_ref(&'a self, offset: usize) -> Self::Item {
        if cfg!(debug_assertions) {
            &self.store[offset]
        } else {
            unsafe { self.store.get_unchecked(offset) }
        }
    }
}

impl<'a, T, Store> GetMut<'a, usize> for Channel<T, Store>
where
    T: 'a,
    Store: DerefMut<Target = [T]>,
{
    type Item = &'a mut T;

    #[inline]
    fn get_mut(&'a mut self, offset: usize) -> Self::Item {
        if cfg!(debug_assertions) {
            &mut self.store[offset]
        } else {
            unsafe { self.store.get_unchecked_mut(offset) }
        }
    }
}

impl<T, Store> GetMutPtr<usize> for Channel<T, Store>
where
    Store: DerefMut<Target = [T]>,
{
    type Item = *mut T;

    #[inline]
    unsafe fn get_mut_ptr(&mut self, offset: usize) -> Self::Item {
        self.store.as_mut_ptr().add(offset)
    }
}

impl_get_via_get_ref_and_clone!(Channel<T, Store>, T, Store);

//  ██████╗ ██████╗ ███╗   ███╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗
// ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║
// ██║     ██║   ██║██╔████╔██║██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║
// ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║
// ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║
//  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct FastChannelsCompression<Chan, B> {
    bytes_compression: B,
    marker: std::marker::PhantomData<Chan>,
}

impl<Chan, B> FastChannelsCompression<Chan, B> {
    pub fn new(bytes_compression: B) -> Self {
        Self {
            bytes_compression,
            marker: Default::default(),
        }
    }

    pub fn bytes_compression(&self) -> &B {
        &self.bytes_compression
    }
}

impl<Chan, B> FromBytesCompression<B> for FastChannelsCompression<Chan, B> {
    fn from_bytes_compression(bytes_compression: B) -> Self {
        Self::new(bytes_compression)
    }
}

/// A compressed `Channel` that decompresses quickly but only on the same platform where it was compressed.
pub struct FastCompressedChannel<T> {
    compressed_bytes: Vec<u8>,
    decompressed_length: usize, // TODO: we should be able to remove this with some refactoring of the Compression trait
    marker: std::marker::PhantomData<T>,
}

impl<T> FastCompressedChannel<T> {
    pub fn compressed_bytes(&self) -> &[u8] {
        &self.compressed_bytes
    }
}

impl<T, B> Compression for FastChannelsCompression<Channel<T>, B>
where
    B: BytesCompression,
    Channel<T>: for<'r> AsRawBytes<'r>,
{
    type Data = Channel<T>;
    type CompressedData = FastCompressedChannel<T>;

    // Compress the map using some `B: BytesCompression`.
    //
    // WARNING: For performance, this reinterprets the inner vector as a byte slice without accounting for endianness. This is
    // not compatible across platforms.
    fn compress(&self, data: &Self::Data) -> Compressed<Self> {
        let mut compressed_bytes = Vec::new();
        self.bytes_compression
            .compress_bytes(&data.as_raw_bytes(), &mut compressed_bytes);

        Compressed::new(FastCompressedChannel {
            compressed_bytes,
            decompressed_length: data.store().len(),
            marker: Default::default(),
        })
    }

    fn decompress(compressed: &Self::CompressedData) -> Self::Data {
        let num_values = compressed.decompressed_length;

        // Allocate the vector with element type T so the alignment is correct.
        let mut decompressed_values: Vec<T> = Vec::with_capacity(num_values);
        unsafe { decompressed_values.set_len(num_values) };
        let mut decompressed_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                decompressed_values.as_mut_ptr() as *mut u8,
                num_values * core::mem::size_of::<T>(),
            )
        };
        B::decompress_bytes(&compressed.compressed_bytes, &mut decompressed_bytes);

        Channel::new(decompressed_values)
    }
}

// ███╗   ███╗██╗   ██╗██╗  ████████╗██╗
// ████╗ ████║██║   ██║██║  ╚══██╔══╝██║
// ██╔████╔██║██║   ██║██║     ██║   ██║
// ██║╚██╔╝██║██║   ██║██║     ██║   ██║
// ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║
// ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝

pub trait Channels {
    type Data;
    type Ptr: MultiMutPtr<Data = Self::Data>;
    type UninitSelf: UninitChannels;
}

pub trait FillChannels: Channels {
    fn fill(value: Self::Data, length: usize) -> Self;
    fn reset_values(&mut self, value: Self::Data);
}

pub trait UninitChannels: Channels {
    type InitSelf;

    unsafe fn maybe_uninit(size: usize) -> Self;
    unsafe fn assume_init(self) -> Self::InitSelf;
}

impl<T> Channels for Channel<T> {
    type Data = T;
    type Ptr = *mut T;
    type UninitSelf = Channel<MaybeUninit<T>>;
}

impl<T> FillChannels for Channel<T>
where
    T: Clone,
{
    fn fill(value: Self::Data, length: usize) -> Self {
        Self::fill(value, length)
    }

    fn reset_values(&mut self, value: Self::Data) {
        self.reset_values(value)
    }
}

impl<T> UninitChannels for Channel<MaybeUninit<T>> {
    type InitSelf = Channel<T>;

    /// Creates an uninitialized channel, mainly for performance.
    /// # Safety
    /// Call `assume_init` after manually initializing all of the values.
    unsafe fn maybe_uninit(size: usize) -> Self {
        let mut store = Vec::with_capacity(size);
        store.set_len(size);

        Channel::new(store)
    }

    /// Transmutes the channel values from `MaybeUninit<T>` to `T` after manual initialization. The implementation just
    /// reconstructs the internal `Vec` after transmuting the data pointer, so the overhead is minimal.
    /// # Safety
    /// All elements of the channel must be initialized.
    unsafe fn assume_init(self) -> Self::InitSelf {
        let transmuted_values = {
            // Ensure the original vector is not dropped.
            let mut v_clone = core::mem::ManuallyDrop::new(self.store);

            Vec::from_raw_parts(
                v_clone.as_mut_ptr() as *mut T,
                v_clone.len(),
                v_clone.capacity(),
            )
        };

        Channel::new(transmuted_values)
    }
}

macro_rules! impl_channels_for_tuple {
    ( $( $var1:ident, $var2:ident : $t:ident ),+ ) => {

        impl<$($t),+> Channels for ($($t,)+)
        where
            $($t: Channels),+
        {
            type Data = ($($t::Data,)+);
            type Ptr = ($(*mut $t::Data,)+);
            type UninitSelf = ($($t::UninitSelf,)+);
        }

        impl<$($t),+> FillChannels for ($($t,)+)
        where
            $($t: FillChannels),+
        {
            fn fill(value: Self::Data, length: usize) -> Self {
                let ($($var1,)+) = value;

                ($($t::fill($var1, length),)+)
            }

            fn reset_values(&mut self, value: Self::Data) {
                let ($($var1,)+) = self;
                let ($($var2,)+) = value;

                $( $var1.reset_values($var2); )+
            }
        }

        impl<$($t),+> UninitChannels for ($($t,)+)
        where
            $($t: UninitChannels),+
        {
            type InitSelf = ($($t::InitSelf,)+);

            unsafe fn maybe_uninit(size: usize) -> Self {
                ($($t::maybe_uninit(size),)+)
            }

            unsafe fn assume_init(self) -> Self::InitSelf {
                let ($($var1,)+) = self;

                ($($t::assume_init($var1),)+)
            }
        }
    }
}

impl_channels_for_tuple! { a1, a2: A }
impl_channels_for_tuple! { a1, a2: A, b1, b2: B }
impl_channels_for_tuple! { a1, a2: A, b1, b2: B, c1, c2: C }
impl_channels_for_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D }
impl_channels_for_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D, e1, e2: E }
impl_channels_for_tuple! { a1, a2: A, b1, b2: B, c1, c2: C, d1, d2: D, e1, e2: E, f1, f2: F }

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    use crate::Get;

    #[test]
    fn tuple_of_channels_can_get() {
        let mut ch1 = Channel::fill(0, 10);
        let mut ch2 = Channel::fill(0, 10);

        assert_eq!((&ch1, &ch2).get(0), (0, 0));
        assert_eq!((&ch1, &ch2).get_ref(0), (&0, &0));
        assert_eq!((&mut ch1, &mut ch2).get_mut(0), (&mut 0, &mut 0));

        let mut owned = (ch1, ch2);

        assert_eq!(owned.get(0), (0, 0));
        assert_eq!(owned.get_ref(0), (&0, &0));
        assert_eq!(owned.get_mut(0), (&mut 0, &mut 0));
    }
}
