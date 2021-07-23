use building_blocks_core::prelude::*;
use building_blocks_mesh::*;
use building_blocks_storage::{prelude::*, IsEmpty};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn greedy_quads_terrace(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_quads_terrace");
    for size in [8, 16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let extent =
                        Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(size)).padded(1);
                    let mut voxels = Array3x1::fill(extent, CubeVoxel(false));
                    for i in 0..size {
                        let level = Extent3i::from_min_and_shape(
                            Point3i::fill(i),
                            PointN([size - i, 1, size - i]),
                        );
                        voxels.fill_extent(&level, CubeVoxel(true));
                    }

                    // Do a single run first to allocate the buffer to the right size.
                    let mut buffer = GreedyQuadsBuffer::new(
                        *voxels.extent(),
                        RIGHT_HANDED_Y_UP_CONFIG.quad_groups(),
                    );
                    greedy_quads(&voxels, voxels.extent(), &mut buffer);

                    (voxels, buffer)
                },
                |(voxels, mut buffer)| greedy_quads(&voxels, voxels.extent(), &mut buffer),
            );
        });
    }
    group.finish();
}

criterion_group!(benches, greedy_quads_terrace);
criterion_main!(benches);

#[derive(Clone)]
struct CubeVoxel(bool);

#[derive(Eq, PartialEq)]
struct TrivialMergeValue;

impl MergeVoxel for CubeVoxel {
    type VoxelValue = TrivialMergeValue;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        TrivialMergeValue
    }
}

impl IsEmpty for CubeVoxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}

impl IsOpaque for CubeVoxel {
    fn is_opaque(&self) -> bool {
        true
    }
}
