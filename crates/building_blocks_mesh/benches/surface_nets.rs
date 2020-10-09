use building_blocks_core::prelude::*;
use building_blocks_mesh::surface_nets::*;
use building_blocks_storage::prelude::*;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn surface_nets_sine_sdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_nets_sine_sdf");
    for radius in [4, 8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(radius), radius, |b, &radius| {
            b.iter_with_setup(
                || {
                    let sample_extent =
                        Extent3i::from_min_and_max(PointN([-radius; 3]), PointN([radius; 3]));
                    let mut samples = Array3::fill(sample_extent, Voxel(0.0));
                    copy_extent(&sample_extent, &sine_sdf, &mut samples);
                    let buffer = SurfaceNetsBuffer::new(sample_extent.num_points());

                    (samples, buffer)
                },
                |(samples, mut buffer)| surface_nets(&samples, samples.extent(), &mut buffer),
            );
        });
    }
    group.finish();
}

criterion_group!(benches, surface_nets_sine_sdf);
criterion_main!(benches);

#[derive(Clone)]
struct Voxel(f32);

impl SignedDistanceVoxel for Voxel {
    fn distance(&self) -> f32 {
        self.0
    }
}

// About the largest radius that can be meshed in a single frame, single-threaded (16.6 ms)
const EXTENT_RADIUS: i32 = 30;

// The higher the frequency (n) the more surface area to mesh.
fn sine_sdf(p: &Point3i) -> Voxel {
    let n = 10.0;
    let val = ((p.x() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin()
        + ((p.y() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin()
        + ((p.z() as f32 / EXTENT_RADIUS as f32) * n * std::f32::consts::PI / 2.0).sin();

    Voxel(val)
}
