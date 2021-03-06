mod compression;
mod serialization;

pub use compression::*;
pub use serialization::*;

use crate::ArrayN;

use building_blocks_core::prelude::*;

/// One piece of a chunked lattice map.
pub trait Chunk<N, T> {
    /// The inner array type. This makes it easier for `Chunk` implementations to satisfy access trait bounds by inheriting them
    /// from existing array types like `ArrayN`.
    type Array;

    /// The value used for vacant space.
    fn ambient_value() -> T;

    /// Construct a new chunk with entirely ambient values.
    fn new_ambient(extent: ExtentN<N>) -> Self;

    /// Borrow the inner array.
    fn array_ref(&self) -> &Self::Array;

    /// Mutably borrow the inner array.
    fn array_mut(&mut self) -> &mut Self::Array;
}

impl<N, T> Chunk<N, T> for ArrayN<N, T>
where
    PointN<N>: IntegerPoint<N>,
    T: Clone + Default,
{
    type Array = Self;

    #[inline]
    fn ambient_value() -> T {
        Default::default()
    }

    #[inline]
    fn new_ambient(extent: ExtentN<N>) -> Self {
        Self::fill(extent, Self::ambient_value())
    }

    #[inline]
    fn array_ref(&self) -> &Self::Array {
        self
    }

    #[inline]
    fn array_mut(&mut self) -> &mut Self::Array {
        self
    }
}
