use building_blocks_core::{point::Point, prelude::*};
use building_blocks_storage::{access::GetUncheckedRefRelease, prelude::*, IsEmpty};

use core::cmp::Ordering;
use core::hash::Hash;
use indexmap::map::Entry::{Occupied, Vacant};
use indexmap::IndexMap;
use std::collections::BinaryHeap;

/// Returns the "surface points" i.e. those points that are non-empty and Von-Neumann-adjacent to an
/// empty point. Since this algorithm does adjacency checks for all points in `extent`, you must
/// ensure that those points are within the bounds of `map`.
pub fn find_surface_points<M, N, T>(map: &M, extent: &ExtentN<N>) -> (Vec<PointN<N>>, Vec<Stride>)
where
    M: Array<N>
        + ForEachRef<N, (PointN<N>, Stride), Data = T>
        + GetRef<Stride, Data = T>
        + GetUncheckedRefRelease<Stride, T>,
    T: IsEmpty,
    PointN<N>: IntegerPoint,
    ExtentN<N>: IntegerExtent<N>,
{
    // Precompute the strides for adjacency checks.
    let vn_offsets: Vec<_> = PointN::von_neumann_offsets()
        .into_iter()
        .map(|p| Local(p))
        .collect();
    let mut vn_strides = vec![Stride(0); vn_offsets.len()];
    map.strides_from_local_points(&vn_offsets, &mut vn_strides);

    let mut surface_points = Vec::new();
    let mut surface_strides = Vec::new();
    map.for_each_ref(&extent, |(p, s), value| {
        if value.is_empty() {
            return;
        }

        for vn_stride in vn_strides.iter() {
            if map.get_unchecked_ref_release(s + *vn_stride).is_empty() {
                surface_points.push(p);
                surface_strides.push(s);
                break;
            }
        }
    });

    (surface_points, surface_strides)
}

/// Uses the given heuristic to do greedy best-first search from `start` to `finish`. All points on
/// the path must satisfy `predicate`. Returns `true` iff the path reaches `finish`. Otherwise,
/// the path that got closest to `finish` is returned after `max_iterations`.
pub fn greedy_path<N, C>(
    start: &PointN<N>,
    finish: &PointN<N>,
    predicate: impl Fn(&PointN<N>) -> bool,
    heuristic: impl Fn(&PointN<N>) -> C,
    max_iterations: usize,
) -> (bool, Vec<PointN<N>>)
where
    C: Copy + Ord,
    PointN<N>: core::hash::Hash + Eq + IntegerPoint,
{
    if !predicate(start) {
        return (false, vec![]);
    }

    let vn_offsets = PointN::<N>::von_neumann_offsets();

    // All adjacent points satisfying predicate.
    let successors = |p: &PointN<N>| {
        vn_offsets
            .iter()
            .map(|offset| *p + *offset)
            .filter(&predicate)
            .collect::<Vec<PointN<N>>>()
    };

    let success = |p: &PointN<N>| *p == *finish;

    let (reached_finish, path) =
        greedy_best_first(start, successors, heuristic, success, max_iterations);

    (reached_finish, path)
}

/// Uses L1 distance as a heuristic to do greedy best-first search from `start` to `finish`. All
/// points on the path must satisfy `predicate`.
pub fn greedy_path_with_l1_heuristic<N>(
    start: &PointN<N>,
    finish: &PointN<N>,
    predicate: impl Fn(&PointN<N>) -> bool,
    max_iterations: usize,
) -> (bool, Vec<PointN<N>>)
where
    PointN<N>: core::hash::Hash + Eq + Distance + IntegerPoint,
    <PointN<N> as Point>::Scalar: Ord,
{
    let heuristic = |p: &PointN<N>| finish.l1_distance(p);

    greedy_path(start, finish, predicate, heuristic, max_iterations)
}

// Some of the "greedy_best_first" code is copied from the "pathfinding" crate and modified to
// support a specific use case. Licensed under dual MIT / Apache 2.0 at the time of copying.

pub fn greedy_best_first<N, C, FN, IN, FH, FS>(
    start: &N,
    mut successors: FN,
    mut heuristic: FH,
    mut success: FS,
    max_iterations: usize,
) -> (bool, Vec<N>)
where
    N: Eq + Hash + Clone,
    C: Ord + Copy,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = N>,
    FH: FnMut(&N) -> C,
    FS: FnMut(&N) -> bool,
{
    let h_start = heuristic(start);
    let mut best_heuristic_so_far = h_start;
    let mut best_index_so_far = 0;

    let mut to_see = BinaryHeap::new();
    to_see.push(HeuristicCostHolder {
        estimated_cost: h_start,
        index: 0,
    });
    let mut parents: IndexMap<N, usize> = IndexMap::new();
    parents.insert(start.clone(), usize::max_value());
    let mut num_iters = 0;
    while let Some(HeuristicCostHolder { index, .. }) = to_see.pop() {
        let successors = {
            let (node, _) = parents.get_index(index).unwrap();
            if success(node) {
                let path = reverse_path(&parents, index);
                return (true, path);
            }

            successors(node)
        };

        for successor in successors {
            let h; // heuristic(&successor)
            let n; // index for successor
            match parents.entry(successor) {
                Vacant(e) => {
                    h = heuristic(e.key());
                    n = e.index();
                    e.insert(index);
                }
                Occupied(_) => {
                    continue;
                }
            }

            to_see.push(HeuristicCostHolder {
                estimated_cost: h,
                index: n,
            });

            if h < best_heuristic_so_far {
                best_heuristic_so_far = h;
                best_index_so_far = n;
            }
        }

        num_iters += 1;
        if num_iters == max_iterations {
            break;
        }
    }

    let path = reverse_path(&parents, best_index_so_far);

    (false, path)
}

fn reverse_path<N>(parents: &IndexMap<N, usize>, start: usize) -> Vec<N>
where
    N: Eq + Hash + Clone,
{
    let path = itertools::unfold(start, |i| {
        parents.get_index(*i).map(|(node, index)| {
            *i = *index;

            node
        })
    })
    .collect::<Vec<&N>>();

    path.into_iter().rev().cloned().collect()
}

struct HeuristicCostHolder<K> {
    estimated_cost: K,
    index: usize,
}

impl<K: PartialEq> PartialEq for HeuristicCostHolder<K> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost)
    }
}

impl<K: PartialEq> Eq for HeuristicCostHolder<K> {}

impl<K: Ord> PartialOrd for HeuristicCostHolder<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for HeuristicCostHolder<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.estimated_cost.cmp(&self.estimated_cost)
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod test {
    use super::*;

    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::iter::FromIterator;

    #[derive(Clone)]
    struct Voxel(bool);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            !self.0
        }
    }

    #[test]
    fn find_surface_points_cube_side_length_3() {
        let mut map = Array3::fill(
            Extent3i::from_min_and_shape(PointN([0; 3]), PointN([5; 3])),
            Voxel(false),
        );

        let solid_extent = Extent3i::from_min_and_shape(PointN([1; 3]), PointN([3; 3]));
        map.for_each_mut(&solid_extent, |_s: Stride, value| *value = Voxel(true));

        // Also set one point on the boundary for an edge case, since it can't be considered, as not
        // all of its neighbors exist.
        *map.get_mut(&PointN([0; 3])) = Voxel(true);

        let (surface_points, _surface_strides) = find_surface_points(&map, &solid_extent);

        // Should exclude the center point.
        let center = PointN([2; 3]);
        let expected_surface_points = solid_extent
            .iter_points()
            .filter(|p| *p != center)
            .collect();
        assert_elements_eq(&surface_points, &expected_surface_points);
    }

    fn assert_elements_eq<T: Clone + Debug + Eq + Hash>(v1: &Vec<T>, v2: &Vec<T>) {
        let set1: HashSet<T> = HashSet::from_iter(v1.iter().cloned());
        let set2: HashSet<T> = HashSet::from_iter(v2.iter().cloned());
        assert_eq!(set1, set2);
    }
}
