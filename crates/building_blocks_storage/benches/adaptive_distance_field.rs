use building_blocks_core::prelude::*;
use building_blocks_storage::{padded_adf_chunk_extent, prelude::*, Adf, AdfVisitNodeId};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn adf_from_array3_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("adf_from_array3_sphere");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || make_sphere_sdf_array(edge_len),
                    |sdf| Adf::from_array3(&sdf, *sdf.extent(), 1.0, 0.1),
                );
            },
        );
    }
    group.finish();
}

fn visit_adf_leaves(c: &mut Criterion) {
    let mut group = c.benchmark_group("visit_adf_leaves");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        let sdf = make_sphere_sdf_array(edge_len);

                        Adf::from_array3(&sdf, *sdf.extent(), 1.0, 0.1)
                    },
                    |adf| {
                        let max_depth = std::usize::MAX;
                        adf.visit_leaves(
                            max_depth,
                            &mut |_node_id, octant, distances: &[f32; 8]| {
                                black_box((octant, distances));
                            },
                        );
                    },
                );
            },
        );
    }
    group.finish();
}

fn visit_adf_minimal_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("visit_adf_minimal_edges");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        let sdf = make_sphere_sdf_array(edge_len);

                        Adf::from_array3(&sdf, *sdf.extent(), 1.0, 0.1)
                    },
                    |adf| {
                        adf.visit_minimal_edges(&mut |nodes: [AdfVisitNodeId; 4]| {
                            black_box(nodes);
                        });
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    adf_from_array3_sphere,
    visit_adf_leaves,
    visit_adf_minimal_edges
);
criterion_main!(benches);

fn make_sphere_sdf_array(edge_length: i32) -> Array3<f32> {
    let sphere_radius = edge_length / 2;
    let center = PointN([0; 3]);

    let extent =
        Extent3i::from_min_and_shape(PointN([-sphere_radius; 3]), PointN([2 * sphere_radius; 3]));
    let padded_extent = padded_adf_chunk_extent(&extent);

    Array3::fill_with(padded_extent, |p| {
        (*p - center).norm() - sphere_radius as f32
    })
}
