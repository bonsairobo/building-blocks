use building_blocks_core::prelude::*;
use building_blocks_partition::octree::Octree;
use building_blocks_storage::{prelude::*, IsEmpty};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn octree_from_array3_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_from_array3_sphere");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &(power, edge_len),
            |b, &(&power, edge_len)| {
                b.iter_with_setup(
                    || {
                        let sphere_radius = edge_len / 2;
                        let mut map = Array3::fill(
                            Extent3i::from_min_and_shape(
                                PointN([-sphere_radius; 3]),
                                PointN([2 * sphere_radius; 3]),
                            ),
                            Voxel(false),
                        );

                        let center = PointN([0; 3]);
                        let map_extent = *map.extent();
                        map.for_each_mut(&map_extent, |p: Point3i, value| {
                            if p.l2_distance_squared(&center) <= sphere_radius * sphere_radius {
                                *value = Voxel(true)
                            }
                        });

                        (map, power)
                    },
                    |(map, power)| Octree::from_array3(power, &map),
                );
            },
        );
    }
    group.finish();
}

fn full_octree(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_octree");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &(power, edge_len),
            |b, &(&power, edge_len)| {
                b.iter_with_setup(
                    || {
                        let map = Array3::fill(
                            Extent3i::from_min_and_shape(PointN([0; 3]), PointN([edge_len; 3])),
                            Voxel(true),
                        );

                        (map, power)
                    },
                    |(map, power)| Octree::from_array3(power, &map),
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, octree_from_array3_sphere, full_octree);
criterion_main!(benches);

#[derive(Clone)]
struct Voxel(bool);

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}
