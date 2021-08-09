use crate::{dot_vox_conversions::VoxColor, prelude::*};

pub use dot_vox;

use building_blocks_core::prelude::*;

use dot_vox::*;

impl Array3x1<VoxColor> {
    pub fn decode_vox(vox_data: &DotVoxData, model_index: usize) -> Self {
        let Model {
            size: Size { x, y, z },
            voxels,
        } = &vox_data.models[model_index];
        let shape = PointN([*x as i32, *y as i32, *z as i32]);
        let extent = Extent3i::from_min_and_shape(PointN([0, 0, 0]), shape);
        let mut map = Array3x1::fill(extent, VoxColor::Empty);
        for Voxel { x, y, z, i } in voxels.iter() {
            let point = PointN([*x as i32, *y as i32, *z as i32]);
            *map.get_mut(point) = VoxColor::Color(*i);
        }

        map
    }
}
