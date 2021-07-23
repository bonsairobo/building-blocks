use crate::{
    array::{
        BorrowChannels, BorrowChannelsMut, Channel, Channels, CopySlices, FastChannelsCompression,
        FillChannels, ResetChannels, Slices, SlicesMut, UninitChannels,
    },
    prelude::Compression,
};

use std::io;

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

        impl<'a, $($t),+> Slices<'a> for ($($t,)+)
        where
            $($t: Slices<'a>),+
        {
            type Target = ($($t::Target,)+);

            fn slices(&'a self) -> Self::Target {
                let ($($var1,)+) = self;

                ($($var1.slices(),)+)
            }
        }

        impl<'a, $($t),+> SlicesMut<'a> for ($($t,)+)
        where
            $($t: SlicesMut<'a>),+
        {
            type Target = ($($t::Target,)+);

            fn slices_mut(&'a mut self) -> Self::Target {
                let ($($var1,)+) = self;

                ($($var1.slices_mut(),)+)
            }
        }

        impl<'a, $($t),+> CopySlices<'a> for ($($t,)+)
        where
            $($t: CopySlices<'a>),+
        {
            type Src = ($($t::Src,)+);

            fn copy_slices(&mut self, src: Self::Src) {
                let ($($var1,)+) = src;
                let ($($var2,)+) = self;

                $( $var2.copy_slices($var1); )+
            }
        }

        impl<'a, $($t),+> BorrowChannels<'a> for ($($t,)+)
        where
            $($t: BorrowChannels<'a>),+
        {
            type Borrowed = ($($t::Borrowed,)+);

            fn borrow(&'a self) -> Self::Borrowed {
                let ($($var1,)+) = self;

                ($($var1.borrow(),)+)
            }
        }

        impl<'a, $($t),+> BorrowChannelsMut<'a> for ($($t,)+)
        where
            $($t: BorrowChannelsMut<'a>),+
        {
            type Borrowed = ($($t::Borrowed,)+);

            fn borrow_mut(&'a mut self) -> Self::Borrowed {
                let ($($var1,)+) = self;

                ($($var1.borrow_mut(),)+)
            }
        }

        impl<$($t),+> FillChannels for ($($t,)+)
        where
            $($t: FillChannels),+
        {
            fn fill(value: Self::Data, length: usize) -> Self {
                let ($($var1,)+) = value;

                ($($t::fill($var1, length),)+)
            }
        }

        impl<$($t),+> ResetChannels for ($($t,)+)
        where
            $($t: ResetChannels),+
        {
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

        impl<$($t),+, By> Compression for FastChannelsCompression<By, ($(Channel<$t>,)+)>
        where
            $( FastChannelsCompression<By, Channel<$t>>: Compression<Data = Channel<$t>>, )+
            By: Clone,
        {
            type Data = ($(Channel<$t>,)+);

            fn compress_to_writer(&self, data: &Self::Data, mut compressed_bytes: impl std::io::Write) -> io::Result<()> {
                let ($($var1,)+) = data;

                // Have to make compression objects for each channel.
                let ($($var2,)+) = ($(FastChannelsCompression::<By, Channel<$t>>::new(self.bytes_compression().clone()),)+);

                // Compress each channel in tuple order.
                $( $var2.compress_to_writer($var1, &mut compressed_bytes)?; )+

                Ok(())
            }

            fn decompress_from_reader(mut compressed_bytes: impl io::Read) -> io::Result<Self::Data> {
                // Decompress each channel in tuple order.
                $( let $var1 = FastChannelsCompression::<By, Channel<$t>>::decompress_from_reader(&mut compressed_bytes)?; )+

                Ok(($($var1,)+))
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

pub mod multichannel_aliases {
    use super::*;

    macro_rules! multichannel_compression_type_alias {
        ($name:ident, $( $t:ident ),+) => {
            pub type $name<By, $($t,)+> = FastChannelsCompression<By, ($(Channel<$t>,)+)>;
        };
    }

    pub type FastChannelsCompression1<By, A> = FastChannelsCompression<By, Channel<A>>;
    multichannel_compression_type_alias!(FastChannelsCompression2, A, B);
    multichannel_compression_type_alias!(FastChannelsCompression3, A, B, C);
    multichannel_compression_type_alias!(FastChannelsCompression4, A, B, C, D);
    multichannel_compression_type_alias!(FastChannelsCompression5, A, B, C, D, E);
    multichannel_compression_type_alias!(FastChannelsCompression6, A, B, C, D, E, F);
}

pub use multichannel_aliases::*;

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    use crate::prelude::{FromBytesCompression, Get, GetMut, GetRef};

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

    #[cfg(feature = "lz4")]
    #[test]
    fn multichannel_compression() {
        use crate::Lz4;

        let channels = (Channel::fill(0, 10), Channel::fill(b'a', 10));

        let compression = FastChannelsCompression2::from_bytes_compression(Lz4 { level: 10 });

        let compressed_channels = compression.compress(&channels);
        let decompressed_channels = compressed_channels.decompress();

        assert_eq!(channels, decompressed_channels);
    }
}
