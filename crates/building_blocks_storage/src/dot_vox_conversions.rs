use crate::prelude::*;

pub use dot_vox;

use building_blocks_core::prelude::*;

use dot_vox::*;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum VoxColor {
    Color(u8),
    Empty,
}

// TODO: should take a type parameter that implements a trait to access `VoxColor`
pub fn encode_vox<Map>(map: &Map, map_extent: Extent3i) -> DotVoxData
where
    Map: for<'r> Get<&'r Point3i, Data = VoxColor>,
{
    let shape = map_extent.shape;
    let vox_extent = map_extent - map_extent.minimum;

    // VOX coordinates are limited to u8.
    assert!(shape <= PointN([std::u8::MAX as i32; 3]));

    let size = dot_vox::Size {
        x: shape.x() as u32,
        y: shape.y() as u32,
        z: shape.z() as u32,
    };

    let mut voxels = Vec::new();
    for (vox_p, map_p) in vox_extent.iter_points().zip(map_extent.iter_points()) {
        if let VoxColor::Color(i) = map.get(&map_p) {
            voxels.push(dot_vox::Voxel {
                x: vox_p.x() as u8,
                y: vox_p.y() as u8,
                z: vox_p.z() as u8,
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
