use building_blocks_core::prelude::*;
use building_blocks_storage::{dev_prelude::*, IsEmpty};

/// Returns the "surface points" i.e. those points that are non-empty and Von-Neumann-adjacent to an empty point. Since this
/// algorithm does adjacency checks for all points in `extent`, you must ensure that those points are within the bounds of
/// `map`.
pub fn find_surface_points<Map, N, T>(
    map: &Map,
    extent: &ExtentN<N>,
) -> (Vec<PointN<N>>, Vec<Stride>)
where
    Map: IndexedArray<N> + ForEach<N, (PointN<N>, Stride), Item = T> + Get<Stride, Item = T>,
    T: IsEmpty,
    PointN<N>: IntegerPoint<N>,
    Local<N>: Copy,
{
    // Precompute the strides for adjacency checks.
    let vn_offsets = Local::localize_points_slice(&PointN::von_neumann_offsets());
    let mut vn_strides = vec![Stride(0); vn_offsets.len()];
    map.strides_from_local_points(&vn_offsets, &mut vn_strides);

    let mut surface_points = Vec::new();
    let mut surface_strides = Vec::new();
    map.for_each(&extent, |(p, s), value| {
        if value.is_empty() {
            return;
        }

        for vn_stride in vn_strides.iter() {
            if map.get(s + *vn_stride).is_empty() {
                surface_points.push(p);
                surface_strides.push(s);
                break;
            }
        }
    });

    (surface_points, surface_strides)
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

    use core::hash::Hash;
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
        let mut map = Array3x1::fill(
            Extent3i::from_min_and_shape(Point3i::ZERO, Point3i::fill(5)),
            Voxel(false),
        );

        let solid_extent = Extent3i::from_min_and_shape(Point3i::fill(1), Point3i::fill(3));
        map.for_each_mut(&solid_extent, |_s: Stride, value| *value = Voxel(true));

        // Also set one point on the boundary for an edge case, since it can't be considered, as not
        // all of its neighbors exist.
        *map.get_mut(Point3i::ZERO) = Voxel(true);

        let (surface_points, _surface_strides) = find_surface_points(&map, &solid_extent);

        // Should exclude the center point.
        let center = Point3i::fill(2);
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
