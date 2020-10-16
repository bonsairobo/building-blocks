//! Constructors for common [signed distance fields](https://en.wikipedia.org/wiki/Signed_distance_function).

use building_blocks_core::prelude::*;

pub fn sphere(center: Point3f, radius: f32) -> impl Fn(Point3i) -> f32 {
    move |p| {
        let pf: Point3f = p.into();

        pf.l2_distance_squared(&center).sqrt() - radius
    }
}

pub fn plane(n: Point3f) -> impl Fn(Point3i) -> f32 {
    move |p| {
        let pf: Point3f = p.into();

        pf.dot(&n)
    }
}

pub fn cube(c: Point3f, r: f32) -> impl Fn(Point3i) -> f32 {
    move |p| {
        let pf: Point3f = p.into();

        let diff = pf - c;
        let max_dim = diff.x().abs().max(diff.y().abs()).max(diff.z().abs());

        max_dim - r
    }
}

pub fn torus(t: Point2f) -> impl Fn(Point3<f32>) -> f32 {
    move |p| {
        let pf: Point3f = p.into();

        let q = PointN([pf.xz().norm() - t.x(), p.y()]);

        q.norm() - t.y()
    }
}
