use crate::prelude::{ChunkKey, ChunkKey2, ChunkKey3};

use building_blocks_core::prelude::*;

use core::ops::RangeInclusive;

pub trait DatabaseKey<N> {
    type Key: Copy + Ord;
    type KeyBytes: AsRef<[u8]>;

    fn into_ord_key(self) -> Self::Key;
    fn from_ord_key(key: Self::Key) -> Self;

    fn ord_key_to_be_bytes(key: Self::Key) -> Self::KeyBytes;
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::Key;

    fn orthant_range(lod: u8, orthant: Orthant<N>) -> RangeInclusive<Self::KeyBytes>;

    fn min_key(lod: u8) -> Self::Key;
    fn max_key(lod: u8) -> Self::Key;
}

impl DatabaseKey<[i32; 2]> for ChunkKey2 {
    type Key = (u8, Morton2);

    // 1 for LOD and 8 for the morton code.
    type KeyBytes = [u8; 9];

    #[inline]
    fn into_ord_key(self) -> Self::Key {
        (self.lod, Morton2::from(self.minimum))
    }

    #[inline]
    fn from_ord_key((lod, morton): Self::Key) -> Self {
        ChunkKey::new(lod, Point2i::from(morton))
    }

    #[inline]
    fn ord_key_to_be_bytes((lod, morton): Self::Key) -> Self::KeyBytes {
        let mut bytes = [0; 9];
        bytes[0] = lod;
        bytes[1..].copy_from_slice(&morton.0.to_be_bytes());

        bytes
    }

    #[inline]
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::Key {
        let lod = bytes[0];
        let mut morton_bytes = [0; 8];
        morton_bytes.copy_from_slice(&bytes[1..]);
        let morton_int = u64::from_be_bytes(morton_bytes);

        (lod, Morton2(morton_int))
    }

    #[inline]
    fn orthant_range(lod: u8, quad: Quadrant) -> RangeInclusive<Self::KeyBytes> {
        let extent = Extent2i::from(quad);
        let min_morton = Morton2::from(extent.minimum);
        let max_morton = Morton2::from(extent.max());
        let min_bytes = Self::ord_key_to_be_bytes((lod, min_morton));
        let max_bytes = Self::ord_key_to_be_bytes((lod, max_morton));

        min_bytes..=max_bytes
    }

    #[inline]
    fn min_key(lod: u8) -> Self::Key {
        (lod, Morton2::from(Point2i::MIN))
    }

    #[inline]
    fn max_key(lod: u8) -> Self::Key {
        (lod, Morton2::from(Point2i::MAX))
    }
}

impl DatabaseKey<[i32; 3]> for ChunkKey3 {
    type Key = (u8, Morton3);

    // 1 for LOD and 12 for the morton code. Although a `Morton3` uses a u128, it only actually uses the least significant 96
    // bits (12 bytes).
    type KeyBytes = [u8; 13];

    #[inline]
    fn into_ord_key(self) -> Self::Key {
        (self.lod, Morton3::from(self.minimum))
    }

    #[inline]
    fn from_ord_key((lod, morton): Self::Key) -> Self {
        ChunkKey::new(lod, Point3i::from(morton))
    }

    #[inline]
    fn ord_key_to_be_bytes((lod, morton): Self::Key) -> Self::KeyBytes {
        let mut bytes = [0; 13];
        bytes[0] = lod;
        bytes[1..].copy_from_slice(&morton.0.to_be_bytes()[4..]);

        bytes
    }

    #[inline]
    fn ord_key_from_be_bytes(bytes: &[u8]) -> Self::Key {
        let lod = bytes[0];
        // The most significant 4 bytes of the u128 are not used.
        let mut morton_bytes = [0; 16];
        morton_bytes[4..16].copy_from_slice(&bytes[1..]);
        let morton_int = u128::from_be_bytes(morton_bytes);

        (lod, Morton3(morton_int))
    }

    #[inline]
    fn orthant_range(lod: u8, octant: Octant) -> RangeInclusive<Self::KeyBytes> {
        let extent = Extent3i::from(octant);
        let min_morton = Morton3::from(extent.minimum);
        let max_morton = Morton3::from(extent.max());
        let min_bytes = Self::ord_key_to_be_bytes((lod, min_morton));
        let max_bytes = Self::ord_key_to_be_bytes((lod, max_morton));

        min_bytes..=max_bytes
    }

    #[inline]
    fn min_key(lod: u8) -> Self::Key {
        (lod, Morton3::from(Point3i::MIN))
    }

    #[inline]
    fn max_key(lod: u8) -> Self::Key {
        (lod, Morton3::from(Point3i::MAX))
    }
}
