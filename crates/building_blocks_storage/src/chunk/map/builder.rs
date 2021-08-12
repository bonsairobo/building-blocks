use crate::{
    array::FillChannels,
    chunk::ChunkNode,
    dev_prelude::{
        Array, Channel, ChunkHashMap, ChunkKey, ChunkMap, ChunkStorage, SmallKeyHashMap,
    },
};

use building_blocks_core::{point_traits::IntegerPoint, ExtentN, PointN};

use core::hash::Hash;
use serde::{Deserialize, Serialize};

/// An object that knows how to construct chunks for a `ChunkMap`.
pub trait ChunkMapBuilder<N, T>: Sized {
    type Chunk;

    fn chunk_shape(&self) -> PointN<N>;

    fn ambient_value(&self) -> T;

    /// Construct a new chunk with entirely ambient values.
    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk;

    /// Create a new `ChunkMap` with the given `storage` which must implement `ChunkStorage`.
    fn build_with_storage<Store>(self, storage: Store) -> ChunkMap<N, T, Self, Store>
    where
        PointN<N>: IntegerPoint<N>,
        Store: ChunkStorage<N, Chunk = ChunkNode<Self::Chunk>>,
    {
        ChunkMap::new(self, storage)
    }

    /// Create a new `ChunkMap` using a `SmallKeyHashMap` as the chunk storage.
    fn build_with_hash_map_storage(self) -> ChunkHashMap<N, T, Self>
    where
        PointN<N>: IntegerPoint<N>,
        ChunkKey<N>: Eq + Hash,
    {
        Self::build_with_storage(self, SmallKeyHashMap::default())
    }
}

/// A `ChunkMapBuilder` for `Array` chunks.
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct ChunkMapBuilderNxM<N, T, Chan> {
    pub chunk_shape: PointN<N>,
    pub ambient_value: T,
    marker: std::marker::PhantomData<Chan>,
}

impl<N, T, Chan> ChunkMapBuilderNxM<N, T, Chan> {
    pub const fn new(chunk_shape: PointN<N>, ambient_value: T) -> Self {
        Self {
            chunk_shape,
            ambient_value,
            marker: std::marker::PhantomData,
        }
    }
}

macro_rules! builder_type_alias {
    ($name:ident, $dim:ty, $( $chan:ident ),+ ) => {
        pub type $name<$( $chan ),+> = ChunkMapBuilderNxM<$dim, ($($chan),+), ($(Channel<$chan>),+)>;
    };
}

pub mod multichannel_aliases {
    use super::*;

    /// A `ChunkMapBuilder` for `ArrayNx1` chunks.
    pub type ChunkMapBuilderNx1<N, A> = ChunkMapBuilderNxM<N, A, Channel<A>>;

    /// A `ChunkMapBuilder` for `Array2x1` chunks.
    pub type ChunkMapBuilder2x1<A> = ChunkMapBuilderNxM<[i32; 2], A, Channel<A>>;
    builder_type_alias!(ChunkMapBuilder2x2, [i32; 2], A, B);
    builder_type_alias!(ChunkMapBuilder2x3, [i32; 2], A, B, C);
    builder_type_alias!(ChunkMapBuilder2x4, [i32; 2], A, B, C, D);
    builder_type_alias!(ChunkMapBuilder2x5, [i32; 2], A, B, C, D, E);
    builder_type_alias!(ChunkMapBuilder2x6, [i32; 2], A, B, C, D, E, F);

    /// A `ChunkMapBuilder` for `Array3x1` chunks.
    pub type ChunkMapBuilder3x1<A> = ChunkMapBuilderNxM<[i32; 3], A, Channel<A>>;
    builder_type_alias!(ChunkMapBuilder3x2, [i32; 3], A, B);
    builder_type_alias!(ChunkMapBuilder3x3, [i32; 3], A, B, C);
    builder_type_alias!(ChunkMapBuilder3x4, [i32; 3], A, B, C, D);
    builder_type_alias!(ChunkMapBuilder3x5, [i32; 3], A, B, C, D, E);
    builder_type_alias!(ChunkMapBuilder3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;

impl<N, T, Chan> ChunkMapBuilder<N, T> for ChunkMapBuilderNxM<N, T, Chan>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone,
    Chan: FillChannels<Data = T>,
{
    type Chunk = Array<N, Chan>;

    fn chunk_shape(&self) -> PointN<N> {
        self.chunk_shape
    }

    fn ambient_value(&self) -> T {
        self.ambient_value.clone()
    }

    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk {
        Array::fill(extent, self.ambient_value())
    }
}
