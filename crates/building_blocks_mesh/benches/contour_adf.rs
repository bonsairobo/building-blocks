use building_blocks_core::prelude::*;
use building_blocks_mesh::{adf_dual_contour, AdfDualContourBuffer};
use building_blocks_storage::{padded_adf_chunk_extent, prelude::*, Adf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn contour_adf(c: &mut Criterion) {
    let mut group = c.benchmark_group("contour_adf");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        let sdf = make_sphere_sdf_array(edge_len);

                        let adf = Adf::from_array3(&sdf, *sdf.extent(), 1.0, 0.2);

                        // Run once just to pre-allocate the buffers.
                        let mut buffer = AdfDualContourBuffer::default();
                        adf_dual_contour(&adf, &mut buffer);

                        (adf, buffer)
                    },
                    |(adf, mut buffer)| {
                        adf_dual_contour(&adf, &mut buffer);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, contour_adf);
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
