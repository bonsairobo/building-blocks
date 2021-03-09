use building_blocks_core::prelude::*;
use building_blocks_search::von_neumann_flood_fill3;
use building_blocks_storage::prelude::*;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn flood_fill_sphere(c: &mut Criterion) {
    let background_color = Color(0);
    let old_color = Color(1);
    let new_color = Color(2);

    let mut group = c.benchmark_group("flood_fill_sphere");
    for radius in [16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(radius),
            radius,
            |b, &sphere_radius| {
                b.iter_with_setup(
                    || {
                        let map_radius = sphere_radius + 1;
                        let mut map = Array3x1::fill(
                            Extent3i::from_min_and_shape(
                                Point3i::fill(-map_radius),
                                Point3i::fill(2 * map_radius),
                            ),
                            background_color,
                        );

                        let center = Point3i::ZERO;
                        let map_extent = *map.extent();
                        map.for_each_mut(&map_extent, |p: Point3i, value| {
                            if p.l2_distance_squared(center) < sphere_radius * sphere_radius {
                                *value = old_color;
                            }
                        });

                        (map, center)
                    },
                    |(mut map, seed)| {
                        let extent = *map.extent();
                        let visitor = |p: Point3i| {
                            if map.get(p) != old_color {
                                return false;
                            }

                            *map.get_mut(p) = new_color;

                            true
                        };
                        von_neumann_flood_fill3(extent, seed, visitor);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, flood_fill_sphere);
criterion_main!(benches);

#[derive(Clone, Copy, Eq, PartialEq)]
struct Color(u8);
