use building_blocks_core::prelude::*;
use building_blocks_mesh::*;
use building_blocks_storage::prelude::*;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn height_map_plane(c: &mut Criterion) {
    let mut group = c.benchmark_group("height_map_plane");
    for size in [8, 16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let sample_extent =
                        Extent2i::from_min_and_max(PointN([0; 2]), PointN([size; 2]));
                    let mut samples = Array2x1::fill(sample_extent, Pixel(0.0));
                    copy_extent(&sample_extent, &Func(plane), &mut samples);

                    // Do a single run first to allocate the buffer to the right size.
                    let mut buffer = HeightMapMeshBuffer::default();
                    triangulate_height_map(&samples, samples.extent(), &mut buffer);

                    (samples, buffer)
                },
                |(samples, mut buffer)| {
                    triangulate_height_map(&samples, samples.extent(), &mut buffer)
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, height_map_plane);
criterion_main!(benches);

fn plane(p: Point2i) -> Pixel {
    Pixel(p.x() as f32 + p.y() as f32)
}

#[derive(Clone)]
struct Pixel(f32);

impl Height for Pixel {
    fn height(&self) -> f32 {
        self.0
    }
}
