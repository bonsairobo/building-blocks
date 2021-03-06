use crate::prelude::*;

pub use image;

use building_blocks_core::prelude::*;

use image::{ImageBuffer, Pixel};

pub fn encode_image<T, P, Map>(
    map: &Map,
    map_extent: Extent2i,
) -> ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>
where
    T: Into<P>,
    Map: Get<Point2i, T>,
    P: Pixel + 'static,
{
    let img_extent = map_extent - map_extent.minimum;
    let shape = img_extent.shape;
    assert!(shape.x() > 0);
    assert!(shape.y() > 0);
    let (width, height) = (shape.x() as u32, shape.y() as u32);

    let mut img = ImageBuffer::new(width, height);
    for (map_p, img_p) in map_extent.iter_points().zip(img_extent.iter_points()) {
        let pixel = map.get(map_p).into();
        *img.get_pixel_mut(img_p.x() as u32, img_p.y() as u32) = pixel;
    }

    img
}
