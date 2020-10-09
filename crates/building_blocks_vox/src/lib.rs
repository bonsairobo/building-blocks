use building_blocks_core::prelude::*;
use building_blocks_storage::{prelude::*, Local};

use dot_vox::*;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum VoxColor {
    Color(u8),
    Empty,
}

pub fn encode_vox(map: &Array3<VoxColor>) -> DotVoxData {
    let global_extent = *map.extent();
    let local_extent = global_extent - global_extent.minimum;
    let shape = local_extent.shape;

    // VOX coordinates are limited to u8.
    assert!(shape <= PointN([std::u8::MAX as i32; 3]));

    let size = dot_vox::Size {
        x: shape.x() as u32,
        y: shape.y() as u32,
        z: shape.z() as u32,
    };

    let mut voxels = Vec::new();
    for p in local_extent.iter_points() {
        if let VoxColor::Color(i) = map.get(&Local(p)) {
            voxels.push(dot_vox::Voxel {
                x: p.x() as u8,
                y: p.y() as u8,
                z: p.z() as u8,
                i,
            });
        }
    }

    let model = dot_vox::Model { size, voxels };

    DotVoxData {
        version: 150,
        models: vec![model],
        palette: Vec::new(),
        materials: Vec::new(),
    }
}

pub fn decode_vox(vox_data: &DotVoxData, model_index: usize) -> Array3<VoxColor> {
    let Model {
        size: Size { x, y, z },
        voxels,
    } = &vox_data.models[model_index];
    let shape = PointN([*x as i32, *y as i32, *z as i32]);
    let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), shape);
    let mut map = Array3::fill(extent, VoxColor::Empty);
    for Voxel { x, y, z, i } in voxels.iter() {
        let point = PointN([*x as i32, *y as i32, *z as i32]);
        *map.get_mut(&point) = VoxColor::Color(*i);
    }

    map
}
