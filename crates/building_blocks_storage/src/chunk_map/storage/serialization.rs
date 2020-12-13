use crate::{
    BincodeCompressedChunk, BincodeCompression, BytesCompression, Chunk, ChunkMap, ChunkShape,
    ChunkWriteStorage, Compression,
};

use building_blocks_core::prelude::*;

use core::hash::Hash;
use fnv::FnvHashMap;
use futures::future::join_all;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// A simple chunk map format for serialization. All chunks are serialized with `bincode`, then compressed using some
/// `BytesCompression`. This can be used to serialize any kind of `ChunkMap`, regardless of `ChunkStorage`.
///
/// This type is provided mostly for convenience. It won't work if your entire `ChunkMap` doesn't fit in memory. That would
/// require a streaming solution.
#[derive(Deserialize, Serialize)]
pub struct SerializableChunkMap<N, T, M, B>
where
    PointN<N>: Hash + Eq,
    Chunk<N, T, M>: DeserializeOwned + Serialize,
    B: BytesCompression,
{
    pub chunk_shape: PointN<N>,
    pub ambient_value: T,
    pub default_chunk_metadata: M,
    pub compressed_chunks: FnvHashMap<PointN<N>, BincodeCompressedChunk<N, T, M, B>>,
}

impl<N, T, M, B> SerializableChunkMap<N, T, M, B>
where
    PointN<N>: IntegerPoint + Hash + Eq + ChunkShape<N>,
    ExtentN<N>: IntegerExtent<N>,
    Chunk<N, T, M>: DeserializeOwned + Serialize,
    T: Copy,
    M: Clone,
    B: BytesCompression,
{
    /// Returns a serializable version of this map. All chunks are serialized with `bincode`, then compressed using some
    /// `BytesCompression`. This can be used to serialize any kind of `ChunkMap`, regardless of `ChunkStorage`.
    pub async fn from_chunk_map<S>(
        params: BincodeCompression<Chunk<N, T, M>, B>,
        map: ChunkMap<N, T, M, S>,
    ) -> Self
    where
        BincodeCompression<Chunk<N, T, M>, B>: Copy, // TODO: this should be inferred
        S: IntoIterator<Item = (PointN<N>, Chunk<N, T, M>)>,
    {
        let chunk_shape = map.indexer.chunk_shape();
        let ambient_value = map.ambient_value();
        let default_chunk_metadata = map.default_chunk_metadata().clone();
        let storage = map.take_storage();

        // Only do one parallel batch at a time to avoid decompressing the entire map at once (assuming the underlying storage
        // does compression).
        let mut compressed_chunks = FnvHashMap::default();
        for batch_of_chunks in &storage.into_iter().chunks(16) {
            for (key, compressed_chunk) in join_all(
                batch_of_chunks
                    .into_iter()
                    .map(|(key, chunk)| async move { (key, params.compress(&chunk)) }),
            )
            .await
            .into_iter()
            {
                compressed_chunks.insert(key, compressed_chunk);
            }
        }

        Self {
            chunk_shape,
            ambient_value,
            default_chunk_metadata,
            compressed_chunks,
        }
    }

    /// Returns a new map from the serialized, compressed version. This will decompress each chunk and insert it into the given
    /// `storage`.
    pub async fn into_chunk_map<S>(self, mut storage: S) -> ChunkMap<N, T, M, S>
    where
        S: ChunkWriteStorage<N, T, M>,
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

        ChunkMap::new(
            self.chunk_shape,
            self.ambient_value,
            self.default_chunk_metadata.clone(),
            storage,
        )
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

        let map = BUILDER.build_with_hash_map_storage();
        let serializable = futures::executor::block_on(SerializableChunkMap::from_chunk_map(
            BincodeCompression::new(Lz4 { level: 10 }),
            map,
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let _deserialized: SerializableChunkMap<[i32; 3], u8, (), Lz4> =
            bincode::deserialize(&serialized).unwrap();
    }

    #[cfg(feature = "snap")]
    #[test]
    fn hash_map_serialize_and_deserialize_round_trip_snappy() {
        use crate::Snappy;

        let map = BUILDER.build_with_hash_map_storage();
        let serializable = futures::executor::block_on(SerializableChunkMap::from_chunk_map(
            BincodeCompression::new(Snappy),
            map,
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let _deserialized: SerializableChunkMap<[i32; 3], u8, (), Snappy> =
            bincode::deserialize(&serialized).unwrap();
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_lz4() {
        use crate::Lz4;

        let map = BUILDER.build(CompressibleChunkStorage::new(Lz4 { level: 10 }));
        let serializable = futures::executor::block_on(SerializableChunkMap::from_chunk_map(
            BincodeCompression::new(Lz4 { level: 10 }),
            map,
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let _deserialized: SerializableChunkMap<[i32; 3], u8, (), Lz4> =
            bincode::deserialize(&serialized).unwrap();
    }

    #[cfg(feature = "snap")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_snappy() {
        use crate::Snappy;

        let map = BUILDER.build(CompressibleChunkStorage::new(Snappy));
        let serializable = futures::executor::block_on(SerializableChunkMap::from_chunk_map(
            BincodeCompression::new(Snappy),
            map,
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let _deserialized: SerializableChunkMap<[i32; 3], u8, (), Snappy> =
            bincode::deserialize(&serialized).unwrap();
    }
}
