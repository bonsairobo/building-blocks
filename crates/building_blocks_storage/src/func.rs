//! Lattice map access traits implemented for functions and closures.
//!
//! This is particularly useful for sampling from signed-distance fields.
//!
//! ```
//! use building_blocks_core::prelude::*;
//! use building_blocks_storage::prelude::*;
//!
//! let sample_extent = Extent3i::from_min_and_max(Point3i::fill(-15), Point3i::fill(15));
//! let mut sampled_sphere = Array3::fill(sample_extent, 0.0);
//!
//! copy_extent(&sample_extent, &|p: Point3i| (p.dot(p) - 10) as f32, &mut sampled_sphere);
//!```

use crate::{ForEach, Get, ReadExtent};

use building_blocks_core::prelude::*;

use core::iter::{once, Once};

impl<F, T, Coord> Get<Coord, T> for F
where
    F: Fn(Coord) -> T,
{
    fn get(&self, c: Coord) -> T {
        (self)(c)
    }
}

impl<F, N, T> ForEach<N, PointN<N>> for F
where
    F: Fn(PointN<N>) -> T,
    PointN<N>: IntegerPoint<N>,
{
    type Item = T;

    fn for_each(&self, extent: &ExtentN<N>, mut f: impl FnMut(PointN<N>, Self::Item)) {
        for p in extent.iter_points() {
            f(p, (self)(p))
        }
    }
}

impl<'a, F, N, T> ReadExtent<'a, N> for F
where
    F: 'a + Fn(PointN<N>) -> T,
    PointN<N>: IntegerPoint<N>,
{
    type Src = &'a Self;
    type SrcIter = Once<(ExtentN<N>, Self::Src)>;

    fn read_extent(&'a self, extent: &ExtentN<N>) -> Self::SrcIter {
        once((*extent, self))
    }
}
