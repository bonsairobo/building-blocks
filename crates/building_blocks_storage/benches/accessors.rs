use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, SmallKeyHashMap};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn array_for_each_stride(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_for_each_stride");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size),
                |(array, iter_extent)| {
                    array.for_each(&iter_extent, |stride: Stride, value| {
                        black_box((stride, value));
                    });
                },
            );
        });
    }
    group.finish();
}

fn array_for_each_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_for_each_point");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size),
                |(array, iter_extent)| {
                    array.for_each(&iter_extent, |p: Point3i, value| {
                        black_box((p, value));
                    });
                },
            );
        });
    }
    group.finish();
}

fn array_for_each_point_and_stride(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_for_each_point_and_stride");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size),
                |(array, iter_extent)| {
                    array.for_each(&iter_extent, |(p, stride): (Point3i, Stride), value| {
                        black_box((p, stride, value));
                    });
                },
            );
        });
    }
    group.finish();
}

fn chunk_hash_map_for_each_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hash_map_for_each_point");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_chunk_map(SmallKeyHashMap::default(), size),
                |(chunk_map, iter_extent)| {
                    chunk_map.for_each(&iter_extent, |p: Point3i, value| {
                        black_box((p, value));
                    });
                },
            );
        });
    }
    group.finish();
}

fn array_point_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_point_indexing");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size),
                |(array, iter_extent)| {
                    for p in iter_extent.iter_points() {
                        black_box(array.get(p));
                    }
                },
            );
        });
    }
    group.finish();
}

fn chunk_hash_map_point_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hash_map_point_indexing");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_chunk_map(SmallKeyHashMap::default(), size),
                |(chunk_map, iter_extent)| {
                    for p in iter_extent.iter_points() {
                        black_box(chunk_map.get(p));
                    }
                },
            );
        });
    }
    group.finish();
}

fn chunk_hash_map_visit_chunks_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hash_map_visit_chunks_sparse");
    for size in [128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_sparse_chunk_map(SmallKeyHashMap::default(), size, 3),
                |(chunk_map, iter_extent)| {
                    chunk_map.visit_occupied_chunks(&iter_extent, |chunk| {
                        black_box(chunk);
                    });
                },
            );
        });
    }
    group.finish();
}

fn compressible_chunk_map_point_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressible_chunk_map_point_indexing");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let storage =
                        FastCompressibleChunkStorage::with_bytes_compression(Lz4 { level: 10 });

                    set_up_chunk_map(storage, size)
                },
                |(chunk_map, iter_extent)| {
                    let local_cache = LocalChunkCache::new();
                    let reader = chunk_map.storage().reader(&local_cache);
                    let reader_map = ChunkMap3x1::build_with_read_storage(CHUNK_SHAPE, reader);
                    for p in iter_extent.iter_points() {
                        black_box(reader_map.get(p));
                    }
                },
            );
        });
    }
    group.finish();
}

fn array_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_copy");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let array_extent =
                        Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(size));
                    let array_src = Array3x1::fill(array_extent, 1);
                    let array_dst = Array3x1::fill(array_extent, 0);

                    let cp_extent = array_extent.padded(-1);

                    (array_src, array_dst, cp_extent)
                },
                |(src, mut dst, cp_extent)| {
                    copy_extent(&cp_extent, &src, &mut dst);
                },
            );
        });
    }
    group.finish();
}

fn chunk_hash_map_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hash_map_copy");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let cp_extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(size));
                    let mut src =
                        ChunkMap3x1::build_with_rw_storage(CHUNK_SHAPE, SmallKeyHashMap::default());
                    src.fill_extent(&cp_extent, 1);

                    let dst =
                        ChunkMap3x1::build_with_rw_storage(CHUNK_SHAPE, SmallKeyHashMap::default());

                    (src, dst, cp_extent)
                },
                |(src, mut dst, cp_extent)| {
                    copy_extent(&cp_extent, &src, &mut dst);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    array_for_each_stride,
    array_for_each_point,
    array_for_each_point_and_stride,
    array_point_indexing,
    array_copy,
    chunk_hash_map_for_each_point,
    chunk_hash_map_point_indexing,
    chunk_hash_map_visit_chunks_sparse,
    chunk_hash_map_copy,
    compressible_chunk_map_point_indexing
);
criterion_main!(benches);

const ARRAY_SIZES: [i32; 3] = [16, 32, 64];

fn set_up_array(size: i32) -> (Array3x1<i32>, Extent3i) {
    let array_extent = Extent3::from_min_and_shape(Point3i::ZERO, Point3i::fill(size));
    let array = Array3x1::fill(array_extent, 1);

    let iter_extent = array_extent.padded(-1);

    (array, iter_extent)
}

fn set_up_chunk_map<Store>(storage: Store, size: i32) -> (ChunkMap3x1<i32, Store>, Extent3i)
where
    Store: ChunkWriteStorage<[i32; 3], Array3x1<i32>>,
{
    let mut map = ChunkMap3x1::build_with_write_storage(CHUNK_SHAPE, storage);
    let iter_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(size));
    map.fill_extent(&iter_extent, 1);

    (map, iter_extent)
}

fn set_up_sparse_chunk_map<Store>(
    storage: Store,
    size: i32,
    sparsity: i32,
) -> (ChunkMap3x1<i32, Store>, Extent3i)
where
    Store: ChunkWriteStorage<[i32; 3], Array3x1<i32>>,
{
    let mut map = ChunkMap3x1::build_with_write_storage(CHUNK_SHAPE, storage);
    let chunk_key_extent =
        Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(size) / CHUNK_SHAPE);
    for chunk_p in chunk_key_extent.iter_points() {
        if chunk_p % sparsity != Point3i::ZERO {
            continue;
        }

        let chunk_key = chunk_p * CHUNK_SHAPE;
        map.write_chunk(
            chunk_key,
            Array3x1::fill(Extent3i::from_min_and_shape(chunk_key, CHUNK_SHAPE), 1),
        );
    }

    let iter_extent = Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(size));

    (map, iter_extent)
}

const CHUNK_SHAPE: Point3i = PointN([16; 3]);
