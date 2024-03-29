use crate::prelude::{ChunkKey, ChunkKey2, ChunkKey3};

use building_blocks_core::prelude::*;

use core::ops::{Bound, RangeInclusive};

pub trait DatabaseKey<N> {
    type OrdKey: Copy + Ord;
    type KeyBytes: AsRef<[u8]>;

    fn into_ord_key(self) -> Self::OrdKey;
    fn from_ord_key(key: Self::OrdKey) -> Self;

    fn ord_key_to_be_bytes(key: Self::OrdKey) -> Self::KeyBytes;
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::OrdKey;

    fn orthant_range(lod: u8, orthant: Orthant<N>) -> RangeInclusive<Self::OrdKey>;

    fn min_key(lod: u8) -> Self::OrdKey;
    fn max_key(lod: u8) -> Self::OrdKey;

    fn full_range(lod: u8) -> RangeInclusive<Self::OrdKey> {
        Self::min_key(lod)..=Self::max_key(lod)
    }
}

impl DatabaseKey<[i32; 2]> for ChunkKey2 {
    type OrdKey = (u8, Morton2);

    // 1 for LOD and 8 for the morton code.
    type KeyBytes = [u8; 9];

    #[inline]
    fn into_ord_key(self) -> Self::OrdKey {
        (self.lod, Morton2::from(self.minimum))
    }

    #[inline]
    fn from_ord_key((lod, morton): Self::OrdKey) -> Self {
        ChunkKey::new(lod, Point2i::from(morton))
    }

    #[inline]
    fn ord_key_to_be_bytes((lod, morton): Self::OrdKey) -> Self::KeyBytes {
        let mut bytes = [0; 9];
        bytes[0] = lod;
        bytes[1..].copy_from_slice(&morton.0.to_be_bytes());
        bytes
    }

    #[inline]
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::OrdKey {
        let lod = bytes[0];
        let mut morton_bytes = [0; 8];
        morton_bytes.copy_from_slice(&bytes[1..]);
        let morton_int = u64::from_be_bytes(morton_bytes);
        (lod, Morton2(morton_int))
    }

    #[inline]
    fn orthant_range(lod: u8, quad: Quadrant) -> RangeInclusive<Self::OrdKey> {
        let extent = Extent2i::from(quad);
        let min_morton = Morton2::from(extent.minimum);
        let max_morton = Morton2::from(extent.max());
        (lod, min_morton)..=(lod, max_morton)
    }

    #[inline]
    fn min_key(lod: u8) -> Self::OrdKey {
        (lod, Morton2::from(Point2i::MIN))
    }

    #[inline]
    fn max_key(lod: u8) -> Self::OrdKey {
        (lod, Morton2::from(Point2i::MAX))
    }
}

impl DatabaseKey<[i32; 3]> for ChunkKey3 {
    type OrdKey = (u8, Morton3);

    // 1 for LOD and 12 for the morton code. Although a `Morton3` uses a u128, it only actually uses the least significant 96
    // bits (12 bytes).
    type KeyBytes = [u8; 13];

    #[inline]
    fn into_ord_key(self) -> Self::OrdKey {
        (self.lod, Morton3::from(self.minimum))
    }

    #[inline]
    fn from_ord_key((lod, morton): Self::OrdKey) -> Self {
        ChunkKey::new(lod, Point3i::from(morton))
    }

    #[inline]
    fn ord_key_to_be_bytes((lod, morton): Self::OrdKey) -> Self::KeyBytes {
        let mut bytes = [0; 13];
        bytes[0] = lod;
        bytes[1..].copy_from_slice(&morton.0.to_be_bytes()[4..]);
        bytes
    }

    #[inline]
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::OrdKey {
        let lod = bytes[0];
        // The most significant 4 bytes of the u128 are not used.
        let mut morton_bytes = [0; 16];
        morton_bytes[4..16].copy_from_slice(&bytes[1..]);
        let morton_int = u128::from_be_bytes(morton_bytes);
        (lod, Morton3(morton_int))
    }

    #[inline]
    fn orthant_range(lod: u8, octant: Octant) -> RangeInclusive<Self::OrdKey> {
        let extent = Extent3i::from(octant);
        let min_morton = Morton3::from(extent.minimum);
        let max_morton = Morton3::from(extent.max());
        (lod, min_morton)..=(lod, max_morton)
    }

    #[inline]
    fn min_key(lod: u8) -> Self::OrdKey {
        (lod, Morton3::from(Point3i::MIN))
    }

    #[inline]
    fn max_key(lod: u8) -> Self::OrdKey {
        (lod, Morton3::from(Point3i::MAX))
    }
}

// TODO: replace this when https://github.com/rust-lang/rust/issues/86026 is stabilized
pub(crate) fn map_bound<X, Y>(b: Bound<&X>, f: impl FnOnce(&X) -> Y) -> Bound<Y> {
    match b {
        Bound::Excluded(x) => Bound::Excluded(f(x)),
        Bound::Included(x) => Bound::Included(f(x)),
        Bound::Unbounded => Bound::Unbounded,
    }
}
