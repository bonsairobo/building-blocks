use building_blocks_core::prelude::*;
use building_blocks_storage::prelude::{
    Array3x1, ChunkDownsampler, ChunkMapBuilder, ChunkMapBuilder3x1, Local, OctreeChunkIndex,
    PointDownsampler, Sd8, SdfMeanDownsampler,
};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn point_downsample3(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_downsample3");
    for size in CHUNK_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let chunk_shape = Point3i::fill(size);
                    let extent = Extent3i::from_min_and_shape(Point3i::ZERO, chunk_shape);
                    let src = Array3x1::fill(extent, 1);
                    let dst = Array3x1::fill(extent, 0);

                    (src, dst, chunk_shape)
                },
                |(src, mut dst, chunk_shape)| {
                    PointDownsampler.downsample(&src, &mut dst, Local(chunk_shape / 2));
                    black_box(dst);
                },
            );
        });
    }
    group.finish();
}

fn sdf_mean_downsample3(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdf_mean_downsample3");
    for size in CHUNK_SIZES.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let chunk_shape = Point3i::fill(size);
                    let extent = Extent3i::from_min_and_shape(Point3i::ZERO, chunk_shape);
                    let src = Array3x1::fill(extent, Sd8::ONE);
                    let dst = Array3x1::fill(extent, Sd8(0));

                    (src, dst, chunk_shape)
                },
                |(src, mut dst, chunk_shape)| {
                    SdfMeanDownsampler.downsample(&src, &mut dst, Local(chunk_shape / 2));
                    black_box(dst);
                },
            );
        });
    }
    group.finish();
}

fn sdf_mean_downsample_chunk_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdf_mean_downsample_chunk_map_with_index");

    let num_lods = 6;
    for map_chunks in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(map_chunks),
            map_chunks,
            |b, &map_chunks| {
                b.iter_with_setup(
                    || {
                        let chunk_exponent = 4;
                        let superchunk_exponent = chunk_exponent + num_lods - 1;
                        let chunk_shape = Point3i::fill(1 << chunk_exponent);

                        let builder = ChunkMapBuilder3x1::new(chunk_shape, Sd8::ONE);
                        let mut map = builder.build_with_hash_map_storage();

                        let map_extent =
                            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(map_chunks))
                                * chunk_shape;
                        map.fill_extent(0, &map_extent, Sd8::NEG_ONE);

                        let index =
                            OctreeChunkIndex::index_chunk_map(superchunk_exponent, num_lods, &map);

                        (map, index, map_extent)
                    },
                    |(mut map, index, map_extent)| {
                        map.downsample_chunks_with_index(&index, &SdfMeanDownsampler, &map_extent)
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    point_downsample3,
    sdf_mean_downsample3,
    sdf_mean_downsample_chunk_map
);
criterion_main!(benches);

const CHUNK_SIZES: [i32; 3] = [16, 32, 64];
