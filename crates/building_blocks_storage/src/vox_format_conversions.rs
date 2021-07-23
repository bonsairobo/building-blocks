use std::{fs::File, io::{Cursor, Read, Seek}, path::Path};

use crate::{
    Array3x1,
    GetMut,
};
use vox_format::{
    data::{VoxBuffer, VoxModelBuffer}, 
    reader::{Error, read_vox_into}, 
    types::{Color, ColorIndex, Palette, Point, Size, Voxel},
};

use building_blocks_core::{Extent3i, Point3i, PointN};
    
/// Re-export `vox_format` crate.
pub use vox_format;


/*

These must live in building_blocks_core`

impl<T> From<Vector<T>> for PointN<[T; 3]> {
    fn from(v: Vector<T>) -> Self {
        PointN(v.into())
    }
}

impl From<Size> for Extent3i {
    fn from(size: Size) -> Self {
        // Note: This can fail, if the component is greater than `i32::MAX`
        Extent3i::from_min_and_shape(
            Default::default(),
            PointN([size.x as i32, size.y as i32, size.z as i32]),
        )
    }
}

impl From<Point> for Point3i {
    fn from(point: Point) -> Self {
        PointN([point.x as i32, point.y as i32, point.z as i32])
    }
}
*/


fn point3i_from_vox(point: Point) -> Point3i {
    PointN([point.x as i32, point.y as i32, point.z as i32])
}

fn extent3i_from_vox(size: Size) -> Extent3i {
    // Note: This can fail, if the component is greater than `i32::MAX`
    Extent3i::from_min_and_shape(
        Default::default(),
        PointN([size.x as i32, size.y as i32, size.z as i32]),
    )
}


impl VoxModelBuffer for Array3x1<ColorIndex> {
    fn new(size: Size) -> Self {
        Array3x1::fill_with(extent3i_from_vox(size), |_point| ColorIndex::default())
    }

    fn set_voxel(&mut self, voxel: Voxel, _palette: &Palette) {
        *self.get_mut(point3i_from_vox(voxel.point)) = voxel.color_index;
    }
}

impl VoxModelBuffer for Array3x1<Color> {
    fn new(size: Size) -> Self {
        Array3x1::fill_with(extent3i_from_vox(size), |_point| Color::default())
    }

    fn set_voxel(&mut self, voxel: Voxel, palette: &Palette) {
        *self.get_mut(point3i_from_vox(voxel.point)) = palette[voxel.color_index];
    }
}



struct ReadSingleModel<T> {
    model_index: usize,
    models_read: usize,
    palette: Palette,
    model: Option<Array3x1<T>>,
}

impl<T> ReadSingleModel<T> {
    pub fn new(model_index: usize) -> Self {
        Self {
            model_index,
            models_read: 0,
            palette: Default::default(),
            model: None,
        }
    }
}

impl<C: Default> VoxBuffer for ReadSingleModel<C>
    where Array3x1<C>: VoxModelBuffer
{
    fn set_model_size(&mut self, model_size: Size) {
        if self.models_read == self.model_index {
            self.model = Some(<Array3x1<C> as VoxModelBuffer>::new(model_size));
        }
        self.models_read += 1;
    }

    fn set_voxel(&mut self, voxel: Voxel) {
        if self.models_read == self.model_index + 1 {
            let model = self.model.as_mut().expect("Expected voxel array to be initialized");
            model.set_voxel(voxel, &self.palette);
        }
    }

    fn set_palette(&mut self, palette: Palette) {
        self.palette = palette;
    }
}


/// Short-hand to read a single model from a reader into an [`crate::Array3x1`].
pub fn from_reader<R: Read + Seek, C: Default>(reader: R, model_index: usize) -> Result<Array3x1<C>, Error>
    where Array3x1<C>: VoxModelBuffer
{
    let mut buffer = ReadSingleModel::new(model_index);
    read_vox_into(reader, &mut buffer)?;

    // TODO: Return error
    let model = buffer.model.unwrap_or_else(|| panic!("VOX file does not contain a model with index {}", model_index));

    Ok(model)    
}

/// Short-hand to read a single model from a byte slice into an [`crate::Array3x1`].
pub fn from_slice<C: Default>(slice: &[u8], model_index: usize) -> Result<Array3x1<C>, Error> 
    where Array3x1<C>: VoxModelBuffer
{
    from_reader(Cursor::new(slice), model_index)
}

/// Short-hand to read a single model from a file into an [`crate::Array3x1`].
pub fn from_file<P: AsRef<Path>, C: Default>(path: P, model_index: usize) -> Result<Array3x1<C>, Error>
    where Array3x1<C>: VoxModelBuffer
{
    from_reader(File::open(path)?, model_index)
}
