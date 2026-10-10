//! Collision worlds: obstacles and their signed distance functions. Batched queries reference
//! worlds by index, so one call can cover thousands of different environments.

use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Obstacle {
    /// Oriented box; `rotation` maps box axes to world axes.
    Cuboid {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Solid cylinder along its local z axis; `rotation` maps local axes to world axes.
    Cylinder {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_height: f32,
    },
    /// Capsule around the local z segment from `-half_length` to `half_length`.
    Capsule {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_length: f32,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct World {
    pub obstacles: Vec<Obstacle>,
}

/// Value returned for clearances when there is nothing to collide with.
pub(crate) const FAR: f32 = 1e30;

impl Obstacle {
    /// Signed distance from `p` to the obstacle surface and its gradient (unit outward direction).
    pub fn distance(&self, p: Vec3) -> (f32, Vec3) {
        match *self {
            Obstacle::Cuboid { center, half_extents, rotation } => {
                let r = Mat3::from_quat(rotation);
                box_distance(r, center, half_extents, p)
            }
            Obstacle::Sphere { center, radius } => sphere_distance(center, radius, p),
            Obstacle::Cylinder { center, rotation, radius, half_height } => {
                cylinder_distance(Mat3::from_quat(rotation), center, radius, half_height, p)
            }
            Obstacle::Capsule { center, rotation, radius, half_length } => {
                capsule_distance(Mat3::from_quat(rotation), center, radius, half_length, p)
            }
        }
    }
}

#[inline]
pub(crate) fn sphere_distance(center: Vec3, radius: f32, p: Vec3) -> (f32, Vec3) {
    let v = p - center;
    let len = v.length();
    (len - radius, if len > 1e-9 { v / len } else { Vec3::Z })
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn box_distance(r: Mat3, center: Vec3, half: Vec3, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let sgn = Vec3::select(lp.cmpge(Vec3::ZERO), Vec3::ONE, Vec3::NEG_ONE);
    let qd = lp.abs() - half;
    let outside = qd.max(Vec3::ZERO);
    let olen = outside.length();
    let (d, gl) = if olen > 0.0 {
        (olen, sgn * outside / olen)
    } else if qd.x >= qd.y && qd.x >= qd.z {
        (qd.x, Vec3::new(sgn.x, 0.0, 0.0))
    } else if qd.y >= qd.z {
        (qd.y, Vec3::new(0.0, sgn.y, 0.0))
    } else {
        (qd.z, Vec3::new(0.0, 0.0, sgn.z))
    };
    (d, r * gl)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn cylinder_distance(r: Mat3, center: Vec3, radius: f32, half_height: f32, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let rho = (lp.x * lp.x + lp.y * lp.y).sqrt();
    let radial = if rho > 1e-9 { Vec3::new(lp.x / rho, lp.y / rho, 0.0) } else { Vec3::X };
    let axial = Vec3::new(0.0, 0.0, if lp.z >= 0.0 { 1.0 } else { -1.0 });
    let (dr, dz) = (rho - radius, lp.z.abs() - half_height);
    let (d, gl) = if dr > 0.0 && dz > 0.0 {
        let d = (dr * dr + dz * dz).sqrt();
        (d, (radial * dr + axial * dz) / d)
    } else if dr >= dz {
        (dr, radial)
    } else {
        (dz, axial)
    };
    (d, r * gl)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn capsule_distance(r: Mat3, center: Vec3, radius: f32, half_length: f32, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let v = lp - Vec3::new(0.0, 0.0, lp.z.clamp(-half_length, half_length));
    let len = v.length();
    (len - radius, r * if len > 1e-9 { v / len } else { Vec3::X })
}
