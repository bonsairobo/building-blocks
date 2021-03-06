use crate::{
    BincodeCompressedChunk, BincodeCompression, BytesCompression, ChunkWriteStorage, Compression,
};

use building_blocks_core::prelude::*;

use futures::future::join_all;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// A simple format for serializing a collection of chunks. All chunks are serialized with `bincode`, then compressed using some
/// `BytesCompression`.
#[derive(Deserialize, Serialize)]
#[serde(bound(deserialize = "Ch: DeserializeOwned"))]
pub struct SerializableChunks<N, Ch, B>
where
    // TODO: try to prevent these bounds which make the struct difficult to use
    N: DeserializeOwned + Serialize,
    Ch: DeserializeOwned + Serialize,
    B: BytesCompression,
{
    pub compressed_chunks: Vec<(PointN<N>, BincodeCompressedChunk<Ch, B>)>,
}

impl<N, Ch, B> SerializableChunks<N, Ch, B>
where
    N: DeserializeOwned + Serialize,
    Ch: DeserializeOwned + Serialize,
    B: BytesCompression,
{
    /// Returns a serializable version of this map. All chunks are serialized with `bincode`, then compressed using some
    /// `BytesCompression`. This can be used to serialize any kind of `ChunkMap`, regardless of chunk storage.
    pub async fn from_iter(
        compression: BincodeCompression<Ch, B>,
        chunks_iter: impl IntoIterator<Item = (PointN<N>, Ch)>,
    ) -> Self
    where
        B: Copy,
    {
        // Only do one parallel batch at a time to avoid decompressing the entire map at once (assuming the underlying storage
        // does compression).
        let mut compressed_chunks = Vec::new();
        for batch_of_chunks in &chunks_iter.into_iter().chunks(16) {
            for (key, compressed_chunk) in join_all(
                batch_of_chunks
                    .into_iter()
                    .map(|(key, chunk)| async move { (key, compression.compress(&chunk)) }),
            )
            .await
            .into_iter()
            {
                compressed_chunks.push((key, compressed_chunk));
            }
        }

        Self { compressed_chunks }
    }

    /// Returns a new map from the serialized, compressed version. This will decompress each chunk and insert it into the given
    /// `storage`.
    pub async fn fill_storage<Store>(self, storage: &mut Store)
    where
        Store: ChunkWriteStorage<N, Ch>,
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

    use fnv::FnvHashMap;

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

        do_serialize_and_deserialize_round_trip_test(FnvHashMap::default(), Snappy);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_lz4() {
        use crate::Lz4;

        let compression = Lz4 { level: 10 };
        do_serialize_and_deserialize_round_trip_test(
            FastCompressibleChunkStorage::with_bytes_compression(compression),
            compression,
        );
    }

    #[cfg(feature = "snap")]
    #[test]
    fn compressible_map_serialize_and_deserialize_round_trip_snappy() {
        use crate::Snappy;

        do_serialize_and_deserialize_round_trip_test(
            FastCompressibleChunkStorage::with_bytes_compression(Snappy),
            Snappy,
        );
    }

    fn do_serialize_and_deserialize_round_trip_test<B, Store>(storage: Store, compression: B)
    where
        Store:
            ChunkWriteStorage<[i32; 3], Array3<i32>> + IntoIterator<Item = (Point3i, Array3<i32>)>,
        B: BytesCompression + Copy + DeserializeOwned + Serialize,
    {
        let mut map = ChunkMap::build_with_write_storage(CHUNK_SHAPE, storage);
        let filled_extent = Extent3i::from_min_and_shape(Point3i::fill(-100), Point3i::fill(200));
        map.fill_extent(&filled_extent, 1);
        let serializable = futures::executor::block_on(SerializableChunks::from_iter(
            BincodeCompression::new(compression),
            map.take_storage(),
        ));
        let serialized: Vec<u8> = bincode::serialize(&serializable).unwrap();
        let deserialized: SerializableChunks<[i32; 3], Array3<i32>, B> =
            bincode::deserialize(&serialized).unwrap();

        let mut storage = FnvHashMap::default();
        futures::executor::block_on(deserialized.fill_storage(&mut storage));
        let map = ChunkMap::build_with_rw_storage(CHUNK_SHAPE, storage);
        map.for_each(&filled_extent, |_p, val| assert_eq!(val, 1));
    }

    const CHUNK_SHAPE: Point3i = PointN([16; 3]);
}
