use building_blocks_core::prelude::*;

// This is the naive implementation. Kept around just for a baseline measurement.
//
// pub fn von_neumann_flood_fill3(bounds: Extent3i, seed: Point3i, mut visitor: impl FnMut(Point3i) -> bool) {
//     let mut stack = vec![seed];
//
//     while let Some(p) = stack.pop() {
//         if bounds.contains(&p) && visitor(p) {
//             stack.push(PointN([p.x() - 1, p.y(), p.z()]));
//             stack.push(PointN([p.x() + 1, p.y(), p.z()]));
//             stack.push(PointN([p.x(), p.y() - 1, p.z()]));
//             stack.push(PointN([p.x(), p.y() + 1, p.z()]));
//             stack.push(PointN([p.x(), p.y(), p.z() - 1]));
//             stack.push(PointN([p.x(), p.y(), p.z() + 1]));
//         }
//     }
// }

/// Visits the Von-Neumann-connected region, starting at `seed`, where all points in the region satisfy `visitor` (i.e.
/// `visitor` returns `true`). The search space is bounded by `bounds`. `visitor` may be called multiple times on the same
/// point, so it must remember which points have been visited already. This is usually accomplished by setting values in an
/// `Array3` that covers the same region as `bounds`.
pub fn von_neumann_flood_fill3(
    bounds: Extent3i,
    seed: Point3i,
    mut visitor: impl FnMut(Point3i) -> bool,
) {
    // This implementation uses scanlines as an optimization over the naive 4-neighbor flood fill. Rather than filling a point
    // and its neighbors on each iteration, we fill a line segment and its neighboring, parallel segments on each iteration.
    // This cuts down on redundant visits.

    // PERF: I think this could be even better using something like "scan planes" or "scan volumes."
    //
    // for radius 32 sphere:
    //
    // naive: 822355 visits
    // scanline: 637318 visits
    // scanplane: ??? visits
    // best case: 3216 visits

    if !visitor(seed) {
        return;
    }

    let bounds_max = bounds.max();

    // A stack of lines already "filled". We still want to extend them in the +/- X directions and scan parallel lines.
    let mut line_stack = vec![ScanLine::new(seed.x(), seed.y(), seed.z())];

    while let Some(l) = line_stack.pop() {
        if l.extend_left {
            // Make a new line segment extending in the -X direction.
            let mut new_x_min = l.x_min;
            while new_x_min > bounds.minimum.x() && visitor(PointN([new_x_min - 1, l.y, l.z])) {
                new_x_min -= 1;
            }
            if new_x_min < l.x_min {
                line_stack.push(ScanLine::new_extended(new_x_min, l.x_min - 1, l.y, l.z));
            }
        }
        if l.extend_right {
            // Make a new line segment extending in the +X direction.
            let mut new_x_max = l.x_max;
            while new_x_max < bounds_max.x() && visitor(PointN([new_x_max + 1, l.y, l.z])) {
                new_x_max += 1;
            }
            if new_x_max > l.x_max {
                line_stack.push(ScanLine::new_extended(l.x_max + 1, new_x_max, l.y, l.z));
            }
        }

        // Visit the 4 parallel line segments, pushing the filled segments onto the stack.
        if l.check_up && l.y < bounds_max.y() {
            let mut parallel = ScanLine::new(l.x_min, l.y + 1, l.z);
            parallel.check_down = false;
            visit_parallel_line(parallel, l.x_max, &mut visitor, &mut line_stack);
        }
        if l.check_down && l.y > bounds.minimum.y() {
            let mut parallel = ScanLine::new(l.x_min, l.y - 1, l.z);
            parallel.check_up = false;
            visit_parallel_line(parallel, l.x_max, &mut visitor, &mut line_stack);
        }
        if l.check_front && l.z < bounds_max.z() {
            let mut parallel = ScanLine::new(l.x_min, l.y, l.z + 1);
            parallel.check_back = false;
            visit_parallel_line(parallel, l.x_max, &mut visitor, &mut line_stack);
        }
        if l.check_back && l.z > bounds.minimum.z() {
            let mut parallel = ScanLine::new(l.x_min, l.y, l.z - 1);
            parallel.check_front = false;
            visit_parallel_line(parallel, l.x_max, &mut visitor, &mut line_stack);
        }
    }
}

fn visit_parallel_line(
    mut l: ScanLine,
    x_max: i32,
    visitor: &mut impl FnMut(Point3i) -> bool,
    line_stack: &mut Vec<ScanLine>,
) {
    l.extend_right = false;
    let mut filling = false;
    for x in l.x_min..=x_max {
        if visitor(PointN([x, l.y, l.z])) {
            if !filling {
                filling = true;
                l.x_min = x;
            }
        } else if filling {
            filling = false;
            l.x_max = x - 1;
            line_stack.push(l.clone());
            l.extend_left = false;
        }
    }
    if filling {
        l.extend_right = true;
        l.x_max = x_max;
        line_stack.push(l);
    }
}

#[derive(Clone)]
struct ScanLine {
    extend_left: bool,
    extend_right: bool,
    check_up: bool,
    check_down: bool,
    check_front: bool,
    check_back: bool,
    x_min: i32,
    x_max: i32,
    y: i32,
    z: i32,
}

impl ScanLine {
    fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            extend_left: true,
            extend_right: true,
            check_up: true,
            check_down: true,
            check_front: true,
            check_back: true,
            x_min: x,
            x_max: x,
            y,
            z,
        }
    }

    fn new_extended(x_min: i32, x_max: i32, y: i32, z: i32) -> Self {
        Self {
            extend_left: false,
            extend_right: false,
            check_up: true,
            check_down: true,
            check_front: true,
            check_back: true,
            x_min,
            x_max,
            y,
            z,
        }
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use super::*;

    use building_blocks_storage::prelude::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Color(u8);

    #[test]
    fn fill_sphere() {
        let background_color = Color(0);
        let old_color = Color(1);
        let new_color = Color(2);

        let sphere_radius = 32;
        let map_radius = sphere_radius + 1;
        let mut map = Array3::fill(
            Extent3i::from_min_and_shape(Point3i::fill(-map_radius), Point3i::fill(2 * map_radius)),
            background_color,
        );

        // Initialize the sphere with "old color."
        let center = Point3i::ZERO;
        let map_extent = *map.extent();
        map.for_each_mut(&map_extent, |p: Point3i, value| {
            if p.l2_distance_squared(center) < sphere_radius * sphere_radius {
                *value = old_color;
            }
        });

        // Flood fill the sphere with "new color."
        let extent = *map.extent();
        let mut num_visits = 0;
        let visitor = |p: Point3i| {
            num_visits += 1;

            if map.get(p) != old_color {
                return false;
            }

            *map.get_mut(p) = new_color;

            true
        };
        von_neumann_flood_fill3(extent, center, visitor);

        // Assert that we actually filled the sphere, and only the sphere.
        map.for_each(&map_extent, |p: Point3i, value| {
            if p.l2_distance_squared(center) < sphere_radius * sphere_radius {
                assert_eq!(value, new_color);
            } else {
                assert_eq!(value, background_color);
            }
        });

        test_print(&format!("# flood fill visits = {}\n", num_visits));
    }

    fn test_print(message: &str) {
        use std::io::Write;

        std::io::stdout()
            .lock()
            .write_all(message.as_bytes())
            .unwrap();
    }
}
