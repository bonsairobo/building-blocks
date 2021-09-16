use crate::{
    array::FillChannels,
    dev_prelude::{Array, Channel, ChunkStorage, ChunkTree, HashMapChunkTree, SmallKeyHashMap},
};

use building_blocks_core::{point_traits::IntegerPoint, ExtentN, PointN};

use core::hash::Hash;
use serde::{Deserialize, Serialize};

/// Constant parameters required to construct a [`ChunkTreeBuilder`].
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct ChunkTreeConfig<N, T> {
    /// The shape of every chunk.
    pub chunk_shape: PointN<N>,
    /// The voxel value taken in regions where chunks are vacant.
    pub ambient_value: T,
    /// The level of detail of root nodes. This implies there are `root_lod + 1` levels of detail, where level 0 (leaves of the
    /// tree) has the highest sample rate.
    pub root_lod: u8,
}

/// An object that knows how to construct chunks for a `ChunkTree`.
pub trait ChunkTreeBuilder<N, T>: Sized {
    type Chunk;

    fn config(&self) -> &ChunkTreeConfig<N, T>;

    /// Construct a new chunk with entirely ambient values.
    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk;

    #[inline]
    fn chunk_shape(&self) -> PointN<N>
    where
        PointN<N>: Clone,
    {
        self.config().chunk_shape.clone()
    }

    #[inline]
    fn ambient_value(&self) -> T
    where
        T: Clone,
    {
        self.config().ambient_value.clone()
    }

    #[inline]
    fn root_lod(&self) -> u8 {
        self.config().root_lod
    }

    #[inline]
    fn num_lods(&self) -> u8 {
        self.root_lod() + 1
    }

    /// Create a new `ChunkTree` with the given `storage` which must implement both `ChunkReadStorage` and `ChunkWriteStorage`.
    fn build_with_storage<Store>(
        self,
        storage_factory: impl Fn() -> Store,
    ) -> ChunkTree<N, T, Self, Store>
    where
        PointN<N>: IntegerPoint,
        T: Clone,
        Store: ChunkStorage<N, Chunk = Self::Chunk>,
    {
        let storages = (0..self.num_lods()).map(|_| storage_factory()).collect();
        ChunkTree::new(self, storages)
    }

    /// Create a new `ChunkTree` using a `SmallKeyHashMap` as the chunk storage.
    fn build_with_hash_map_storage(self) -> HashMapChunkTree<N, T, Self>
    where
        PointN<N>: Hash + IntegerPoint,
        T: Clone,
    {
        Self::build_with_storage(self, SmallKeyHashMap::default)
    }
}

/// A `ChunkTreeBuilder` for `Array` chunks.
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct ChunkTreeBuilderNxM<N, T, Chan> {
    pub config: ChunkTreeConfig<N, T>,
    marker: std::marker::PhantomData<Chan>,
}

impl<N, T, Chan> ChunkTreeBuilderNxM<N, T, Chan> {
    pub const fn new(config: ChunkTreeConfig<N, T>) -> Self {
        Self {
            config,
            marker: std::marker::PhantomData,
        }
    }
}

macro_rules! builder_type_alias {
    ($name:ident, $dim:ty, $( $chan:ident ),+ ) => {
        pub type $name<$( $chan ),+> = ChunkTreeBuilderNxM<$dim, ($($chan),+), ($(Channel<$chan>),+)>;
    };
}

pub mod multichannel_aliases {
    use super::*;

    /// A `ChunkTreeBuilder` for `ArrayNx1` chunks.
    pub type ChunkTreeBuilderNx1<N, A> = ChunkTreeBuilderNxM<N, A, Channel<A>>;

    /// A `ChunkTreeBuilder` for `Array2x1` chunks.
    pub type ChunkTreeBuilder2x1<A> = ChunkTreeBuilderNxM<[i32; 2], A, Channel<A>>;
    builder_type_alias!(ChunkTreeBuilder2x2, [i32; 2], A, B);
    builder_type_alias!(ChunkTreeBuilder2x3, [i32; 2], A, B, C);
    builder_type_alias!(ChunkTreeBuilder2x4, [i32; 2], A, B, C, D);
    builder_type_alias!(ChunkTreeBuilder2x5, [i32; 2], A, B, C, D, E);
    builder_type_alias!(ChunkTreeBuilder2x6, [i32; 2], A, B, C, D, E, F);

    /// A `ChunkTreeBuilder` for `Array3x1` chunks.
    pub type ChunkTreeBuilder3x1<A> = ChunkTreeBuilderNxM<[i32; 3], A, Channel<A>>;
    builder_type_alias!(ChunkTreeBuilder3x2, [i32; 3], A, B);
    builder_type_alias!(ChunkTreeBuilder3x3, [i32; 3], A, B, C);
    builder_type_alias!(ChunkTreeBuilder3x4, [i32; 3], A, B, C, D);
    builder_type_alias!(ChunkTreeBuilder3x5, [i32; 3], A, B, C, D, E);
    builder_type_alias!(ChunkTreeBuilder3x6, [i32; 3], A, B, C, D, E, F);
}

pub use multichannel_aliases::*;

impl<N, T, Chan> ChunkTreeBuilder<N, T> for ChunkTreeBuilderNxM<N, T, Chan>
where
    PointN<N>: IntegerPoint,
    T: Clone,
    Chan: FillChannels<Data = T>,
{
    type Chunk = Array<N, Chan>;

    fn config(&self) -> &ChunkTreeConfig<N, T> {
        &self.config
    }

    fn new_ambient(&self, extent: ExtentN<N>) -> Self::Chunk {
        Array::fill(extent, self.ambient_value())
    }
}
