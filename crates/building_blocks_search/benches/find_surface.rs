use building_blocks_core::prelude::*;
use building_blocks_search::surface::find_surface_points;
use building_blocks_storage::{prelude::*, IsEmpty};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn sphere_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere_surface");
    for radius in [8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(radius),
            radius,
            |b, &sphere_radius| {
                b.iter_with_setup(
                    || {
                        let map_radius = sphere_radius + 1;
                        let mut map = Array3::fill(
                            Extent3i::from_min_and_shape(
                                Point3i::fill(-map_radius),
                                Point3i::fill(2 * map_radius),
                            ),
                            Voxel(false),
                        );

                        let center = Point3i::ZERO;
                        let map_extent = *map.extent();
                        map.for_each_mut(&map_extent, |p: Point3i, value| {
                            if p.l2_distance_squared(center) < sphere_radius * sphere_radius {
                                *value = Voxel(true)
                            }
                        });

                        map
                    },
                    |map| find_surface_points(&map, &map.extent().padded(-1)),
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, sphere_surface);
criterion_main!(benches);

#[derive(Clone)]
struct Voxel(bool);

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}
