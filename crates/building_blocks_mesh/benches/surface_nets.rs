use building_blocks_core::prelude::*;
use building_blocks_mesh::*;
use building_blocks_storage::prelude::*;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn surface_nets_sine_sdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_nets_sine_sdf");
    for diameter in [8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(diameter),
            diameter,
            |b, &diameter| {
                b.iter_with_setup(
                    || {
                        let radius = diameter >> 1;
                        let sample_extent = Extent3i::from_min_and_max(
                            Point3i::fill(-radius),
                            Point3i::fill(radius),
                        );
                        let mut samples = Array3x1::fill(sample_extent, Sd8(0));
                        copy_extent(&sample_extent, &Func(sine_sdf), &mut samples);

                        // Do a single run first to allocate the buffer to the right size.
                        let mut buffer = SurfaceNetsBuffer::default();
                        surface_nets(&samples, samples.extent(), 1.0, true, &mut buffer);

                        (samples, buffer)
                    },
                    |(samples, mut buffer)| {
                        surface_nets(&samples, samples.extent(), 1.0, true, &mut buffer)
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, surface_nets_sine_sdf);
criterion_main!(benches);

// About the largest radius that can be meshed in a single frame, single-threaded (16.6 ms)
const EXTENT_RADIUS: i32 = 30;

// The higher the frequency (n) the more surface area to mesh.
fn sine_sdf(p: Point3i) -> Sd8 {
    let n = 10.0;
    let val = ((p.x() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin()
        + ((p.y() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin()
        + ((p.z() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin();

    Sd8::from(val)
}
