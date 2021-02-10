use building_blocks_core::prelude::*;
use building_blocks_storage::{Array3, ChunkDownsampler, Local, PointDownsampler};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn point_downsample3(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_downsample3");
    for size in CHUNK_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let chunk_shape = PointN([size; 3]);
                    let extent = Extent3i::from_min_and_shape(Point3i::ZERO, chunk_shape);
                    let src = Array3::fill(extent, 1);
                    let dst = Array3::fill(extent, 0);

                    (src, dst, chunk_shape)
                },
                |(src, mut dst, chunk_shape)| {
                    PointDownsampler.downsample(&src, &mut dst, Local(chunk_shape / 2), 1);
                    black_box(dst);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, point_downsample3);
criterion_main!(benches);

const CHUNK_SIZES: [i32; 3] = [16, 32, 64];
