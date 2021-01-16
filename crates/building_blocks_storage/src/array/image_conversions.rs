use crate::{prelude::*, Local};

use building_blocks_core::prelude::*;

use core::mem::MaybeUninit;
use image::GenericImageView;

impl<Im> From<&Im> for Array2<<Im as GenericImageView>::Pixel>
where
    Im: GenericImageView,
{
    fn from(image: &Im) -> Self {
        let shape = PointN([image.width() as i32, image.height() as i32]);
        let extent = Extent2i::from_min_and_shape(Point2i::ZERO, shape);
        let mut map: Array2<MaybeUninit<<Im as GenericImageView>::Pixel>> =
            unsafe { Array2::maybe_uninit(extent) };
        for (x, y, pixel) in image.pixels() {
            let point = PointN([x as i32, y as i32]);
            unsafe {
                map.get_mut(&Local(point)).as_mut_ptr().write(pixel);
            }
        }

        unsafe { map.assume_init() }
    }
}
