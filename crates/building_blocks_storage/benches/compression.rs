use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, BincodeLz4};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn decompress_array_with_bincode_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_bincode_lz4");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size).compress(BincodeLz4 { level: 10 }),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

fn decompress_array_with_fast_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_fast_lz4");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || set_up_array(size).compress(FastLz4 { level: 10 }),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    decompress_array_with_bincode_lz4,
    decompress_array_with_fast_lz4
);
criterion_main!(benches);

const ARRAY_SIZES: [i32; 3] = [16, 32, 64];

fn set_up_array(size: i32) -> Array3<i32> {
    let array_extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([size; 3]));

    // Might be tough to compress this.
    Array3::fill_with(array_extent, |p: &Point3i| {
        p.x() % 3 + p.y() % 3 + p.z() % 3
    })
}

// TODO: report the compression efficiency for some typical data set, like a sphere
