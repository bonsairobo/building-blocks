use crate::{
    BincodeCompressedChunk, BincodeCompression, BytesCompression, Chunk, ChunkMap, ChunkMapBuilder,
    ChunkShape, ChunkWriteStorage, Compression,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use futures::future::join_all;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// A simple chunk map format for serialization. All chunks are serialized with `bincode`, then compressed using some
/// `BytesCompression`. This can be used to:
///   - serialize any `ChunkMap` whose storage implements `IntoIterator<Item = (PointN<N>, Chunk<N, T, Meta>)>`
///   - deserialize any `ChunkMap` whose storage implements `ChunkWriteStorage`
///
/// This type is provided mostly for convenience. It won't work if your entire `ChunkMap` doesn't fit in memory. That would
/// require a streaming solution.
#[derive(Deserialize, Serialize)]
pub struct SerializableChunkMap<N, T, Meta, B>
where
    PointN<N>: Hash + Eq,
    Chunk<N, T, Meta>: DeserializeOwned + Serialize,
    B: BytesCompression,
{
    pub builder: ChunkMapBuilder<N, T, Meta>,
    pub compressed_chunks: FnvHashMap<PointN<N>, BincodeCompressedChunk<N, T, Meta, B>>,
}

impl<N, T, Meta, B> SerializableChunkMap<N, T, Meta, B>
where
    PointN<N>: IntegerPoint<N> + Hash + Eq + ChunkShape<N>,
    Chunk<N, T, Meta>: DeserializeOwned + Serialize,
    T: Copy,
    Meta: Clone,
    B: BytesCompression,
{
    /// Returns a serializable version of this map. All chunks are serialized with `bincode`, then compressed using some
    /// `BytesCompression`. This can be used to serialize any kind of `ChunkMap`, regardless of chunk storage.
    pub async fn from_chunk_map<Store>(
        compression: BincodeCompression<Chunk<N, T, Meta>, B>,
        map: ChunkMap<N, T, Meta, Store>,
    ) -> Self
    where
        BincodeCompression<Chunk<N, T, Meta>, B>: Copy, // TODO: this should be inferred
        Store: IntoIterator<Item = (PointN<N>, Chunk<N, T, Meta>)>,
    {
        let builder = map.builder();
        let storage = map.take_storage();

        // Only do one parallel batch at a time to avoid decompressing the entire map at once (assuming the underlying storage
        // does compression).
        let mut compressed_chunks = FnvHashMap::default();
        for batch_of_chunks in &storage.into_iter().chunks(16) {
            for (key, compressed_chunk) in join_all(
                batch_of_chunks
                    .into_iter()
                    .map(|(key, chunk)| async move { (key, compression.compress(&chunk)) }),
            )
            .await
            .into_iter()
            {
                compressed_chunks.insert(key, compressed_chunk);
            }
        }

        Self {
            builder,
            compressed_chunks,
        }
    }

    /// Returns a new map from the serialized, compressed version. This will decompress each chunk and insert it into the given
    /// `storage`.
    pub async fn into_chunk_map<Store>(self, mut storage: Store) -> ChunkMap<N, T, Meta, Store>
    where
        Store: ChunkWriteStorage<N, T, Meta>,
    {
        // Only do one parallel batch at a time to avoid decompressing the entire map at once.
        for batch_of_compressed_chunks in &self.compressed_chunks.into_iter().chunks(16) {
            for (key, chunk) in
                join_all(batch_of_compressed_chunks.into_iter().map(
                    |(key, compressed_chunk)| async move { (key, compressed_chunk.decompress()) },
                ))
                .await
                .into_iter()
            {
                storage.write(key, chunk);
            }
        }

        self.builder.build_with_write_storage(storage)
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    use crate::prelude::*;

    const BUILDER: ChunkMapBuilder<[i32; 3], i32, ()> = ChunkMapBuilder {
        chunk_shape: PointN([16; 3]),
        ambient_value: 0,
        default_chunk_metadata: (),
    };

    #[cfg(feature = "lz4")]
    #[test]
    fn hash_map_serialize_and_deserialize_round_trip_lz4() {
        use crate::Lz4;

        let compression = Lz4 { level: 10 };
        do_serialize_and_deserialize_round_trip_test(FnvHashMap::default(), compression);
    }

    #[cfg(feature = "snap")]
    #[test]
    fn hash_map_serialize_and_deserialize_round_trip_snappy() {
        use crate::Snappy;

        let compression = Snappy;
        do_serialize_and_deserialize_round_trip_test(FnvHashMap::default(), compression);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_lz4() {
        use crate::Lz4;

        let compression = Lz4 { level: 10 };
        do_serialize_and_deserialize_round_trip_test(
            CompressibleChunkStorage::new(compression),
            compression,
        );
    }

    #[cfg(feature = "snap")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_snappy() {
        use crate::Snappy;

        let compression = Snappy;
        do_serialize_and_deserialize_round_trip_test(
            CompressibleChunkStorage::new(compression),
            compression,
        );
    }

    fn do_serialize_and_deserialize_round_trip_test<B, Store>(storage: Store, compression: B)
    where
        Store:
            ChunkWriteStorage<[i32; 3], i32, ()> + IntoIterator<Item = (Point3i, Chunk3<i32, ()>)>,
        B: BytesCompression + Copy + DeserializeOwned + Serialize,
    {
        let mut map = BUILDER.build_with_write_storage(storage);
        let filled_extent = Extent3i::from_min_and_shape(PointN([-100; 3]), PointN([200; 3]));
        map.fill_extent(&filled_extent, 1);
        let serializable = futures::executor::block_on(SerializableChunkMap::from_chunk_map(
            BincodeCompression::new(compression),
            map,
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let deserialized: SerializableChunkMap<[i32; 3], i32, (), B> =
            bincode::deserialize(&serialized).unwrap();

        let map = futures::executor::block_on(deserialized.into_chunk_map(FnvHashMap::default()));
        map.for_each(&filled_extent, |_p, val| assert_eq!(val, 1));
    }
}
