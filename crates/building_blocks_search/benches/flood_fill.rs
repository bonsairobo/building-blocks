use building_blocks_core::prelude::*;
use building_blocks_search::von_neumann_flood_fill3;
use building_blocks_storage::prelude::*;

use utilities::data_sets::sphere_bit_array;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn flood_fill_sphere(c: &mut Criterion) {
    let background_color = Color(0);
    let old_color = Color(1);
    let new_color = Color(2);

    let mut group = c.benchmark_group("flood_fill_sphere");
    for array_edge_length in [16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(array_edge_length),
            array_edge_length,
            |b, &array_edge_length| {
                b.iter_with_setup(
                    || {
                        (
                            sphere_bit_array(array_edge_length, old_color, background_color).0,
                            Point3i::ZERO,
                        )
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
