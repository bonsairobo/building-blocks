use building_blocks_core::prelude::*;
use building_blocks_storage::{
    access_traits::*,
    database::{ChunkDb3, Delta, ReadableChunkDb},
    prelude::{
        ChunkKey, ChunkTreeBuilder, ChunkTreeBuilder3x1, ChunkTreeConfig, FastArrayCompressionNx1,
        FromBytesCompression, Lz4,
    },
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn db_read_all_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("db_read_all_chunks");

    for map_chunks in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(map_chunks),
            map_chunks,
            |b, &map_chunks| {
                b.iter_with_setup(
                    || {
                        let chunk_exponent = 4;
                        let chunk_shape = Point3i::fill(1 << chunk_exponent);

                        let builder = ChunkTreeBuilder3x1::new(ChunkTreeConfig {
                            chunk_shape,
                            ambient_value: 1,
                            root_lod: 0,
                        });
                        let mut map = builder.build_with_hash_map_storage();

                        let map_extent =
                            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(map_chunks))
                                * chunk_shape;
                        map.lod_view_mut(0)
                            .for_each_mut(&map_extent, |p: Point3i, d| {
                                *d = p.x() % 3 + p.y() % 3 + p.z() % 3
                            });

                        let db = sled::Config::default().temporary(true).open().unwrap();
                        let tree = db.open_tree("test").unwrap();
                        let chunk_db = ChunkDb3::new_with_compression(
                            tree,
                            FastArrayCompressionNx1::from_bytes_compression(Lz4 { level: 10 }),
                        );

                        let mut batch = chunk_db.start_delta_batch();
                        futures::executor::block_on(
                            batch.add_and_compress_deltas(
                                map.take_storages().pop().unwrap().into_iter().filter_map(
                                    |(k, v)| {
                                        v.user_chunk.map(|u| Delta::Insert(ChunkKey::new(0, k), u))
                                    },
                                ),
                            ),
                        );

                        chunk_db.apply_deltas(batch.build()).unwrap();
                        futures::executor::block_on(chunk_db.flush()).unwrap();

                        chunk_db
                    },
                    |chunk_db| {
                        let _result = chunk_db.read_all_chunks::<[i32; 3]>(0).unwrap();
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, db_read_all_chunks);
criterion_main!(benches);
