use building_blocks_core::prelude::*;
use building_blocks_storage::Array3x1;

// TODO: it would be nice if all crates could share this module, but it causes this issue:
// https://github.com/rust-lang/cargo/issues/6765

pub fn sphere_bit_array<T>(
    array_edge_length: i32,
    inner_value: T,
    outer_value: T,
) -> (Array3x1<T>, i32)
where
    T: Copy,
{
    let array_radius = array_edge_length / 2;
    let sphere_radius = array_radius - 1;
    let array_extent = Extent3i::from_min_and_shape(
        Point3i::fill(-array_radius),
        Point3i::fill(array_edge_length),
    );

    let map = Array3x1::fill_with(array_extent, |p| {
        if p.norm() < sphere_radius as f32 {
            inner_value
        } else {
            outer_value
        }
    });

    (map, sphere_radius)
}
