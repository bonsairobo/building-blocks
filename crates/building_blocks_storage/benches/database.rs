use building_blocks_core::prelude::*;
use building_blocks_storage::{
    access_traits::*,
    database::{ChunkDb3, Delta},
    prelude::{
        ChunkMapBuilder, ChunkMapBuilder3x1, FastArrayCompressionNx1, FromBytesCompression, Lz4,
    },
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn db_write_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("db_write_chunks");

    for map_chunks in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(map_chunks),
            map_chunks,
            |b, &map_chunks| {
                b.iter_with_setup(
                    || {
                        let chunk_exponent = 4;
                        let chunk_shape = Point3i::fill(1 << chunk_exponent);

                        let builder = ChunkMapBuilder3x1::new(chunk_shape, 1);
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
                        let chunk_db = ChunkDb3::new(
                            tree,
                            FastArrayCompressionNx1::from_bytes_compression(Lz4 { level: 10 }),
                        );

                        let mut batch = chunk_db.start_delta_batch();
                        futures::executor::block_on(
                            batch.add_deltas(
                                map.take_storage()
                                    .into_iter()
                                    .map(|(k, v)| Delta::Insert(k, v)),
                            ),
                        );

                        (chunk_db, batch.build())
                    },
                    |(chunk_db, batch)| {
                        chunk_db.apply_deltas(batch).unwrap();
                        futures::executor::block_on(chunk_db.flush()).unwrap();
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, db_write_chunks);
criterion_main!(benches);
