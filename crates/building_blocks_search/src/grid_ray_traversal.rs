use building_blocks_core::prelude::*;

/// Grid traversal algorithm by Amanatides and Woo. Visits every voxel intersecting the given ray.
pub struct GridRayTraversal<Ni, Nf> {
    // The current pixel/voxel position.
    current_grid_point: PointN<Ni>,
    // Either -1 or +1 in each axis. The direction we step along each axis.
    step: PointN<Ni>,
    // The amount of time it takes to move 1 unit along each axis.
    t_delta: PointN<Nf>,
    // The next time when each axis will cross a pixel boundary.
    t_max: PointN<Nf>,
}

/// 2D `GridRayTraversal`.
pub type GridRayTraversal2 = GridRayTraversal<[i32; 2], [f32; 2]>;
/// 3D `GridRayTraversal`.
pub type GridRayTraversal3 = GridRayTraversal<[i32; 3], [f32; 3]>;

impl<Ni, Nf> GridRayTraversal<Ni, Nf> {
    /// Initialize the traversal, beginning at the `start` position and moving along the `velocity` vector.
    #[inline]
    pub fn new(start: PointN<Nf>, velocity: PointN<Nf>) -> Self
    where
        PointN<Ni>: IntegerPoint<Ni>,
        PointN<Nf>: IntoIntegerPoint<IntPoint = PointN<Ni>> + FloatPoint<Nf>,
        PointN<Nf>: From<PointN<Ni>>,
    {
        let current_grid_point: PointN<Ni> = start.into_int();
        let vel_signs = velocity.signum();
        let step = vel_signs.into_int();
        let t_delta = vel_signs / velocity;

        // For each axis, calculate the time delta we need to reach a pixel boundary on that axis. For a positive velocity, this
        // is just the next pixel, but for negative, it's the current pixel (hence the join with zero).
        let next_bounds: PointN<Nf> = (current_grid_point + step.join(PointN::ZERO)).into();
        let delta_to_next_bounds = next_bounds - start;
        let t_max = delta_to_next_bounds / velocity;

        Self {
            current_grid_point,
            step,
            t_delta,
            t_max,
        }
    }
}

impl GridRayTraversal2 {
    /// Move to the next closest pixel along the ray.
    #[inline]
    pub fn step(&mut self) {
        if self.t_max.x() < self.t_max.y() {
            *self.current_grid_point.x_mut() += self.step.x();
            *self.t_max.x_mut() += self.t_delta.x();
        } else {
            *self.current_grid_point.y_mut() += self.step.y();
            *self.t_max.y_mut() += self.t_delta.y();
        }
    }

    /// The current pixel position. Changes on every call of `step`.
    #[inline]
    pub fn current_pixel(&self) -> Point2i {
        self.current_grid_point
    }
}

impl GridRayTraversal3 {
    /// Move the the next closest voxel along the ray.
    #[inline]
    pub fn step(&mut self) {
        if self.t_max.x() < self.t_max.y() {
            if self.t_max.x() < self.t_max.z() {
                *self.current_grid_point.x_mut() += self.step.x();
                *self.t_max.x_mut() += self.t_delta.x();
            } else {
                *self.current_grid_point.z_mut() += self.step.z();
                *self.t_max.z_mut() += self.t_delta.z();
            }
        } else if self.t_max.y() < self.t_max.z() {
            *self.current_grid_point.y_mut() += self.step.y();
            *self.t_max.y_mut() += self.t_delta.y();
        } else {
            *self.current_grid_point.z_mut() += self.step.z();
            *self.t_max.z_mut() += self.t_delta.z();
        }
    }

    /// The current voxel position. Changes on every call of `step`.
    #[inline]
    pub fn current_voxel(&self) -> Point3i {
        self.current_grid_point
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

    #[test]
    fn test_move_along_x_axis() {
        let mut traversal =
            GridRayTraversal3::new(PointN([0.5, 0.5, 0.5]), PointN([1.0, 0.0, 0.0]));

        let mut voxels = Vec::new();
        for _ in 0..5 {
            voxels.push(traversal.current_voxel());
            traversal.step();
        }

        assert_eq!(
            voxels,
            vec![
                PointN([0, 0, 0]),
                PointN([1, 0, 0]),
                PointN([2, 0, 0]),
                PointN([3, 0, 0]),
                PointN([4, 0, 0])
            ]
        )
    }

    #[test]
    fn test_move_along_all_axes_some_negative() {
        let mut traversal =
            GridRayTraversal3::new(PointN([0.5, 0.5, 0.5]), PointN([1.0, -2.0, 3.0]));

        let mut voxels = Vec::new();
        for _ in 0..10 {
            voxels.push(traversal.current_voxel());
            traversal.step();
        }

        assert_eq!(
            voxels,
            vec![
                PointN([0, 0, 0]),
                PointN([0, 0, 1]),
                PointN([0, -1, 1]),
                PointN([0, -1, 2]),
                PointN([1, -1, 2]),
                PointN([1, -2, 2]),
                PointN([1, -2, 3]),
                PointN([1, -2, 4]),
                PointN([1, -3, 4]),
                PointN([2, -3, 4]),
            ]
        )
    }
}
