use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, BincodeCompression};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "lz4")]
use building_blocks_storage::Lz4;
#[cfg(feature = "snap")]
use building_blocks_storage::Snappy;

#[cfg(feature = "lz4")]
fn decompress_array_with_bincode_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_bincode_lz4");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || BincodeCompression::new(Lz4 { level: 10 }).compress(&set_up_array(size)),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

#[cfg(feature = "lz4")]
fn decompress_array_with_fast_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_fast_lz4");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || FastArrayCompression::new(Lz4 { level: 10 }).compress(&set_up_array(size)),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

#[cfg(feature = "snap")]
fn decompress_array_with_bincode_snappy(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_bincode_snappy");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || BincodeCompression::new(Snappy).compress(&set_up_array(size)),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

#[cfg(feature = "snap")]
fn decompress_array_with_fast_snappy(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_array_with_fast_snappy");
    for size in ARRAY_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || FastArrayCompression::new(Snappy).compress(&set_up_array(size)),
                |compressed_array| {
                    compressed_array.decompress();
                },
            );
        });
    }
    group.finish();
}

#[cfg(feature = "lz4")]
criterion_group!(
    lz4_benches,
    decompress_array_with_bincode_lz4,
    decompress_array_with_fast_lz4,
);
#[cfg(feature = "snap")]
criterion_group!(
    snappy_benches,
    decompress_array_with_bincode_snappy,
    decompress_array_with_fast_snappy,
);
#[cfg(all(not(feature = "lz4"), feature = "snap"))]
criterion_main!(snappy_benches);
#[cfg(all(feature = "lz4", not(feature = "snap")))]
criterion_main!(lz4_benches);
#[cfg(all(feature = "lz4", feature = "snap"))]
criterion_main!(lz4_benches, snappy_benches);

const ARRAY_SIZES: [i32; 3] = [16, 32, 64];

fn set_up_array(size: i32) -> Array3<i32> {
    let array_extent = Extent3::from_min_and_shape(PointN([0; 3]), PointN([size; 3]));

    // Might be tough to compress this.
    Array3::fill_with(array_extent, |p: Point3i| p.x() % 3 + p.y() % 3 + p.z() % 3)
}
