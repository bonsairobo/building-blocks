use crate::{
    Channel, Channels, Compressed, Compression, FastChannelsCompression, FillChannels,
    UninitChannels,
};

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

        impl<$($t),+, By> Compression for FastChannelsCompression<By, ($(Channel<$t>,)+)>
        where
            $( FastChannelsCompression<By, Channel<$t>>: Compression<Data = Channel<$t>>, )+
            By: Clone,
        {
            type Data = ($(Channel<$t>,)+);
            type CompressedData = ($(Compressed<FastChannelsCompression<By, Channel<$t>>>,)+);

            fn compress(&self, data: &Self::Data) -> Compressed<Self> {
                let ($($var1,)+) = data;

                // Have to make compression objects for each channel.
                let ($($var2,)+) = ($(FastChannelsCompression::<By, Channel<$t>>::new(self.bytes_compression().clone()),)+);

                Compressed::new(($($var2.compress($var1),)+))
            }

            fn decompress(compressed: &Self::CompressedData) -> Self::Data {
                let ($($var1,)+) = compressed;

                ( $($var1.decompress(),)+ )
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

    use crate::{FromBytesCompression, Get, GetMut, GetRef};

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

        let channels = (Channel::fill(0, 10), Channel::fill('a', 10));

        let compression = FastChannelsCompression2::from_bytes_compression(Lz4 { level: 10 });

        let compressed_channels = compression.compress(&channels);
        let decompressed_channels = compressed_channels.decompress();

        assert_eq!(channels, decompressed_channels);
    }
}
