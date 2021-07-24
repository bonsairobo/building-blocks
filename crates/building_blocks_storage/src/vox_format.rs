//! Provides implementations for loading [MagicaVoxel `.VOX`][magica_voxel] files with the [`vox-format`] crate:
//!
//! 1. [`VoxModelBuffer`] implementions for [`Array3x1`] for all channel types that implement [`VoxChannel`].
//! 1. [`VoxChannel`] implementations for [`ColorIndex`] and [`Color`].
//! 1. Utility functions [`from_reader`], [`from_slice`], and [`from_file`] to read a model from a `.VOX`
//!    file with a single function call.
//!
//! If you want to load into an [`Array`] with other channel types, you can easily implement
//! [`VoxChannel`] yourself:
//!
//! ```rust
//! use vox_format::types::{Voxel, Palette, ColorIndex};
//! use building_blocks_core::Point3i;
//! use building_blocks_storage::{array::Array3x1, vox_format::{VoxChannel, from_file}};
//!
//! pub enum MyChannel {
//!   Air,
//!   Dirt,
//!   Stone,
//! }
//!
//! impl Default for MyChannel {
//!   fn default() -> Self {
//!     Self::Air
//!   }
//! }
//!
//! impl VoxChannel for MyChannel {
//!   fn from_vox(color_index: ColorIndex, _palette: &Palette) -> Self {
//!     // Return your own channel type, given the color index. You can use the palette to lookup the color of this
//!     // voxel, if you want to use the color information.
//!
//!     match color_index.into() {
//!       // This is unnecessary, since color index 0 is always the empty voxel, and will not even be stored in the
//!       // file. Thus this method is also never called with that color index. Instead `MyChannel` should should
//!       // return empty voxels via it's `Default` implementation.
//!       0 => Self::Air,
//!       // Color index 0 is hard-coded to be dirt.
//!       1 => Self::Dirt,
//!       // Color index 1 is hard-coded to be stone.
//!       2 => Self::Stone,
//!       // All other color indices are unused. We just return Air here.
//!       _ => Self::Air,
//!     }
//!   }
//! }
//!
//! // Load the voxels from the first model in a file.
//! # let path = "../../examples/assets/test_single_model_default_palette.vox";
//! let array = from_file::<MyChannel, _>(path, 0)
//!   .expect("reading file failed")
//!   .expect("no model found");
//!
//! ```
//!
//! # Note
//!
//! The implementations that load models into `Array3x1<ColorIndex>` and `Array3x1<Color>` always set a color index
//! or color, even if the voxel is not specified in the file. If no voxel is specified in the file the color index 0
//! will be used, which is fixed by MagicaVoxel to be fully transparent.
//!
//! [`vox-format`]: https://docs.rs/vox-format
//! [magica_voxel]: https://ephtracy.github.io/
//! [`Array`]: crate::array::Array

use std::{
    fs::File,
    io::{Cursor, Read, Seek},
    path::Path,
};

use vox_format::{
    data::{VoxBuffer, VoxModelBuffer},
    reader::{read_vox_into, Error},
    types::{Color, ColorIndex, Palette, Size, Voxel},
};

use building_blocks_core::Point3i;

/// Re-export of the `vox-format` crate.
pub use vox_format;

use crate::{access_traits::GetMut, array::Array3x1};

/// Trait that defines how channel values are read from `.VOX` voxels.
pub trait VoxChannel {
    fn from_vox(color_index: ColorIndex, palette: &Palette) -> Self;
}

impl VoxChannel for ColorIndex {
    fn from_vox(color_index: ColorIndex, _palette: &Palette) -> Self {
        color_index
    }
}

impl VoxChannel for Color {
    fn from_vox(color_index: ColorIndex, palette: &Palette) -> Self {
        palette[color_index]
    }
}

impl<C: VoxChannel + Default> VoxModelBuffer for Array3x1<C> {
    fn new(size: Size) -> Self {
        Array3x1::fill_with(size.into(), |_point| C::default())
    }

    fn set_voxel(&mut self, voxel: Voxel, palette: &Palette) {
        let point = Point3i::from(voxel.point);
        *self.get_mut(point) = C::from_vox(voxel.color_index, palette);
    }
}

/// Reads a single model from a `.VOX` file into an `Array3x1`.
///
/// # Note
///
/// This is private, because this functionality will be implemented in the next release of `vox-format`.
///
struct ReadSingleModel<C> {
    model_index: usize,
    models_read: usize,
    palette: Palette,
    model: Option<Array3x1<C>>,
}

impl<C> ReadSingleModel<C> {
    pub fn new(model_index: usize) -> Self {
        Self {
            model_index,
            models_read: 0,
            palette: Default::default(),
            model: None,
        }
    }
}

impl<C> VoxBuffer for ReadSingleModel<C>
where
    Array3x1<C>: VoxModelBuffer,
{
    fn set_model_size(&mut self, model_size: Size) {
        if self.models_read == self.model_index {
            self.model = Some(<Array3x1<C> as VoxModelBuffer>::new(model_size));
        }
        self.models_read += 1;
    }

    fn set_voxel(&mut self, voxel: Voxel) {
        if self.models_read == self.model_index + 1 {
            let model = self
                .model
                .as_mut()
                .expect("Expected voxel array to be initialized");
            model.set_voxel(voxel, &self.palette);
        }
    }

    fn set_palette(&mut self, palette: Palette) {
        self.palette = palette;
    }
}

/// Short-hand to read a single model from a reader into an [`crate::Array3x1`].
pub fn from_reader<C, R: Read + Seek>(
    reader: R,
    model_index: usize,
) -> Result<Option<Array3x1<C>>, Error>
where
    Array3x1<C>: VoxModelBuffer,
{
    let mut buffer = ReadSingleModel::new(model_index);
    read_vox_into(reader, &mut buffer)?;
    Ok(buffer.model)
}

/// Short-hand to read a single model from a byte slice into an [`crate::Array3x1`].
pub fn from_slice<C>(slice: &[u8], model_index: usize) -> Result<Option<Array3x1<C>>, Error>
where
    Array3x1<C>: VoxModelBuffer,
{
    from_reader(Cursor::new(slice), model_index)
}

/// Short-hand to read a single model from a file into an [`crate::Array3x1`].
///
/// This example loads a `Array3x1<ChannelIndex>` from a vox file:
///
/// ```rust
/// use vox_format::types::ColorIndex;
/// use building_blocks_storage::vox_format::from_file;
/// # let path = "../../examples/assets/test_single_model_default_palette.vox";
/// let model_number = 0;
/// let array = from_file::<ColorIndex, _>(path, model_number)
///   .expect("reading file failed")
///   .expect("no model found");
/// ```
pub fn from_file<C, P: AsRef<Path>>(
    path: P,
    model_index: usize,
) -> Result<Option<Array3x1<C>>, Error>
where
    Array3x1<C>: VoxModelBuffer,
{
    from_reader(File::open(path)?, model_index)
}

#[cfg(test)]
mod tests {
    use building_blocks_core::Point3i;
    use vox_format::types::ColorIndex;

    use crate::access_traits::ForEach;

    use super::from_slice;

    #[test]
    fn it_reads_a_single_model_into_an_array() {
        let data = include_bytes!("../../../examples/assets/test_single_model_default_palette.vox");
        let array = from_slice::<ColorIndex>(data, 0).unwrap().unwrap();

        array.for_each(array.extent(), |point: Point3i, value| {
            let expected = match point.0 {
                [0, 0, 1] | [1, 0, 0] | [2, 0, 0] => ColorIndex::from(79),
                [2, 0, 1] | [2, 0, 2] => ColorIndex::from(69),
                _ => ColorIndex::default(),
            };
            assert_eq!(ColorIndex::from(expected), value);
        });
    }
}
