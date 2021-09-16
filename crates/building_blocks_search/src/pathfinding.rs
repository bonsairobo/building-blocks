use building_blocks_core::{num::Zero, point_traits::Point, itertools, prelude::*};

use core::cmp::Ordering;
use core::hash::Hash;
use indexmap::map::Entry::{Occupied, Vacant};
use indexmap::IndexMap;
use pathfinding::directed::astar::astar;
use std::collections::BinaryHeap;

/// Uses the given heuristic to do an a-star search from `start` to `finish`. All points on the path must satisfy
/// `predicate`. Returns `Some` iff the path reaches `finish`. Otherwise, `None` is returned. The `predicate` must
/// return the cost of moving from the node to the successor if the successor is valid. The `heuristic` function
/// must not return a cost greater than the real cost.
pub fn astar_path<N, C>(
    start: PointN<N>,
    finish: PointN<N>,
    predicate: impl Fn(&PointN<N>) -> Option<C>,
    heuristic: impl Fn(&PointN<N>) -> C,
) -> Option<(Vec<PointN<N>>, C)>
where
    C: Zero + Copy + Ord,
    PointN<N>: core::hash::Hash + Eq + IntegerPoint,
{
    predicate(&start)?;

    let vn_offsets = PointN::<N>::von_neumann_offsets();

    let successors = |p: &PointN<N>| {
        vn_offsets
            .iter()
            .map(|offset| *p + *offset)
            .filter_map(|p| predicate(&p).map(|c| (p, c)))
            .collect::<Vec<(PointN<N>, C)>>()
    };

    let success = |p: &PointN<N>| *p == finish;

    astar(&start, successors, heuristic, success)
}

/// Uses the given heuristic to do greedy best-first search from `start` to `finish`. All points on the path must satisfy
/// `predicate`. Returns `true` iff the path reaches `finish`. Otherwise, the path that got closest to `finish` is returned
/// after `max_iterations`.
pub fn greedy_path<N, C>(
    start: PointN<N>,
    finish: PointN<N>,
    predicate: impl Fn(&PointN<N>) -> bool,
    heuristic: impl Fn(&PointN<N>) -> C,
    max_iterations: usize,
) -> (bool, Vec<PointN<N>>)
where
    C: Copy + Ord,
    PointN<N>: core::hash::Hash + Eq + IntegerPoint,
{
    if !predicate(&start) {
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

    let success = |p: &PointN<N>| *p == finish;

    let (reached_finish, path) =
        greedy_best_first(&start, successors, heuristic, success, max_iterations);

    (reached_finish, path)
}

/// Uses L1 distance as a heuristic to do greedy best-first search from `start` to `finish`. All points on the path must satisfy
/// `predicate`.
pub fn greedy_path_with_l1_heuristic<N>(
    start: PointN<N>,
    finish: PointN<N>,
    predicate: impl Fn(&PointN<N>) -> bool,
    max_iterations: usize,
) -> (bool, Vec<PointN<N>>)
where
    PointN<N>: core::hash::Hash + Eq + Distance + IntegerPoint,
    <PointN<N> as Point>::Scalar: Ord,
{
    let heuristic = |&p: &PointN<N>| finish.l1_distance(p);

    greedy_path(start, finish, predicate, heuristic, max_iterations)
}

// Some of the "greedy_best_first" code is copied from the "pathfinding" crate and modified to support a specific use case.
// Licensed under dual MIT / Apache 2.0 at the time of copying.

/// A generic greedy best first search. You define the graph structure using a `successors` function. If you decide to stop
/// after `max_iterations`, then the node with the least heuristic will be returned.
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
