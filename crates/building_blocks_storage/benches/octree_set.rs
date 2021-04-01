use building_blocks_core::prelude::*;
use building_blocks_storage::{
    octree_set::{OctreeNode, OctreeSet, VisitStatus},
    prelude::*,
    IsEmpty,
};
use utilities::data_sets::sphere_bit_array;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn octree_from_array3_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_from_array3_sphere");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || sphere_bit_array(edge_len, Voxel(true), Voxel(false)).0,
                    |map| OctreeSet::from_array3(&map, *map.extent()),
                );
            },
        );
    }
    group.finish();
}

fn octree_from_array3_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_from_array3_full");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        Array3x1::fill(
                            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(edge_len)),
                            Voxel(true),
                        )
                    },
                    |map| OctreeSet::from_array3(&map, *map.extent()),
                );
            },
        );
    }
    group.finish();
}

fn octree_visit_branches_and_fat_leaves_of_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_visit_branches_and_fat_leaves_of_sphere");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        let map = sphere_bit_array(edge_len, Voxel(true), Voxel(false)).0;

                        OctreeSet::from_array3(&map, *map.extent())
                    },
                    |octree| {
                        octree.visit_branches_and_fat_leaves_in_preorder(
                            &mut |node: &OctreeNode| {
                                black_box(node);

                                VisitStatus::Continue
                            },
                        )
                    },
                );
            },
        );
    }
    group.finish();
}

fn octree_visit_branch_and_leaf_nodes_of_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_visit_branch_and_leaf_nodes_of_sphere");
    for power in [4, 5, 6].iter() {
        let edge_len = 1 << *power;
        group.bench_with_input(
            BenchmarkId::from_parameter(edge_len),
            &edge_len,
            |b, &edge_len| {
                b.iter_with_setup(
                    || {
                        let map = sphere_bit_array(edge_len, Voxel(true), Voxel(false)).0;

                        OctreeSet::from_array3(&map, *map.extent())
                    },
                    |octree| {
                        let mut queue = vec![octree.root_node()];
                        while !queue.is_empty() {
                            if let Some(node) = queue.pop().unwrap() {
                                black_box(&node);
                                if !node.is_full() {
                                    for octant in 0..8 {
                                        queue.push(octree.get_child(&node, octant));
                                    }
                                }
                            }
                        }
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    octree_from_array3_sphere,
    octree_from_array3_full,
    octree_visit_branches_and_fat_leaves_of_sphere,
    octree_visit_branch_and_leaf_nodes_of_sphere
);
criterion_main!(benches);

#[derive(Clone, Copy)]
struct Voxel(bool);

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}
