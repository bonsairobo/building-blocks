use building_blocks_search::find_surface_points;
use building_blocks_storage::IsEmpty;

use utilities::data_sets::sphere_bit_array;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn sphere_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere_surface");
    for array_edge_length in [8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(array_edge_length),
            array_edge_length,
            |b, &array_edge_length| {
                b.iter_with_setup(
                    || sphere_bit_array(array_edge_length, Voxel(true), Voxel(false)).0,
                    |map| find_surface_points(&map, &map.extent().padded(-1)),
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, sphere_surface);
criterion_main!(benches);

#[derive(Clone, Copy)]
struct Voxel(bool);

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}
