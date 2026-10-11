//! Camera images of worlds and the robot in them: a ray caster over the obstacles (exact for the
//! primitives, sphere tracing for distance grids) and the robot's collision spheres. Each pixel
//! gets a depth and a colour: one per obstacle (by its index in the world) and one for the robot,
//! shaded by a light from above. `render.wgsl` mirrors the ray casting here.

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::error::{Result, ensure_input, input};
use crate::robot::Robot;
use crate::sdf::{Intrinsics, SdfGrid, grid_distance};
use crate::types::Pose;
use crate::world::{Obstacle, World};

/// A pinhole camera (OpenCV axes: x right, y down, z forward) and where it is.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Names the camera's features in exports (`observation.images.<name>`).
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub intrinsics: Intrinsics,
    pub mount: Mount,
}

#[derive(Clone, Debug)]
pub enum Mount {
    /// Fixed in the world at this pose.
    World(Pose),
    /// On a robot link, at `offset` in the link's frame (a wrist camera).
    Link { link: String, offset: Pose },
}

/// Rendered views: `[views, height, width, 3]` RGB and `[views, height, width]` depth along each
/// camera's z axis in meters, 0 where nothing is hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Images {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub depth: Vec<f32>,
}

/// One view as the renderers draw it: the world, the camera's pose, the robot's spheres in the
/// world (centre, radius) and an obstacle moved to a pose.
pub(crate) struct View {
    pub(crate) world: u32,
    pub(crate) eye: Pose,
    pub(crate) spheres: Vec<[f32; 4]>,
    pub(crate) moved: Option<(usize, Pose)>,
}

impl Camera {
    pub(crate) fn check(&self, robot: &Robot) -> Result<()> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        ensure_input!(self.width > 0 && self.height > 0, "camera '{}' needs a positive image size", self.name);
        ensure_input!(
            fx > 0.0 && fy > 0.0 && cx.is_finite() && cy.is_finite(),
            "camera '{}' has invalid intrinsics {:?}",
            self.name,
            self.intrinsics
        );
        if let Mount::Link { link, .. } = &self.mount {
            ensure_input!(
                robot.link_pose(robot.default_q(), link).is_some(),
                "camera '{}' is on unknown link '{link}'",
                self.name
            );
        }
        Ok(())
    }

    /// The views of configurations `q` (`[items, dof]`) in `item_world`, with obstacles moved.
    pub(crate) fn views(
        &self,
        robot: &Robot,
        worlds: &[World],
        item_world: &[u32],
        q: &[f32],
        moved: &[Option<(usize, Pose)>],
    ) -> Result<Vec<View>> {
        self.check(robot)?;
        let n = robot.dof();
        ensure_input!(
            moved.is_empty() || moved.len() == item_world.len(),
            "{} moved obstacles for {} views",
            moved.len(),
            item_world.len()
        );
        item_world
            .iter()
            .enumerate()
            .map(|(i, &world)| {
                let q = &q[i * n..(i + 1) * n];
                let moved = moved.get(i).copied().flatten();
                if let Some((k, _)) = moved {
                    let count = worlds[world as usize].obstacles.len();
                    ensure_input!(k < count, "view {i} moves obstacle {k} of {count}");
                }
                let eye = match &self.mount {
                    Mount::World(pose) => *pose,
                    Mount::Link { link, offset } => {
                        robot.link_pose(q, link).ok_or_else(|| input!("unknown link '{link}'"))?.mul_pose(*offset)
                    }
                };
                let fk = robot.fk(q);
                let spheres = robot
                    .spheres
                    .iter()
                    .map(|s| (fk.rot[s.link] * s.center + fk.pos[s.link]).extend(s.radius).to_array())
                    .collect();
                Ok(View { world, eye, spheres, moved })
            })
            .collect()
    }
}

/// Colour of each obstacle by its index in the world, then of the robot.
pub(crate) const PALETTE: [[f32; 3]; 8] = [
    [0.55, 0.42, 0.30],
    [0.20, 0.45, 0.80],
    [0.85, 0.35, 0.25],
    [0.30, 0.70, 0.35],
    [0.85, 0.70, 0.20],
    [0.55, 0.35, 0.70],
    [0.25, 0.70, 0.70],
    [0.80, 0.45, 0.60],
];
pub(crate) const ROBOT: [f32; 3] = [0.78, 0.78, 0.80];
pub(crate) const BACKGROUND: [f32; 3] = [0.10, 0.10, 0.12];

/// Draws `views` on the CPU, each pixel independently.
pub(crate) fn render(camera: &Camera, worlds: &[World], views: &[View]) -> Images {
    let (w, h) = (camera.width as usize, camera.height as usize);
    let mut rgb = vec![0u8; views.len() * w * h * 3];
    let mut depth = vec![0.0f32; views.len() * w * h];
    rgb.par_chunks_mut(w * 3).zip(depth.par_chunks_mut(w)).enumerate().for_each(|(row, (rgb, depth))| {
        let (view, v) = (&views[row / h], row % h);
        let to_world = Mat3::from_quat(view.eye.rotation);
        for u in 0..w {
            let k = camera.intrinsics;
            let dir = to_world * Vec3::new((u as f32 - k.cx) / k.fx, (v as f32 - k.cy) / k.fy, 1.0);
            let (t, color) = shade(&worlds[view.world as usize], view, view.eye.position, dir);
            depth[u] = t;
            rgb[u * 3..u * 3 + 3].copy_from_slice(&color.map(|c| (c * 255.0).round() as u8));
        }
    });
    Images { width: camera.width, height: camera.height, rgb, depth }
}

/// The nearest hit along `origin + t * dir` (`dir` has unit camera-z length, so `t` is depth) and
/// its shaded colour; `(0, BACKGROUND)` without one.
fn shade(world: &World, view: &View, origin: Vec3, dir: Vec3) -> (f32, [f32; 3]) {
    let mut best = (f32::INFINITY, Vec3::Z, ROBOT);
    for s in &view.spheres {
        if let Some(t) = ray_sphere(origin, dir, Vec3::new(s[0], s[1], s[2]), s[3])
            && t < best.0
        {
            best = (t, (origin + dir * t - Vec3::new(s[0], s[1], s[2])) / s[3], ROBOT);
        }
    }
    for (k, o) in world.obstacles.iter().enumerate() {
        let moved;
        let o = match view.moved {
            Some((m, pose)) if m == k => {
                moved = o.placed(pose);
                &moved
            }
            _ => o,
        };
        if let Some((t, normal)) = intersect(o, origin, dir, best.0) {
            best = (t, normal, PALETTE[k % PALETTE.len()]);
        }
    }
    if !best.0.is_finite() {
        return (0.0, BACKGROUND);
    }
    let light = Vec3::new(0.3, 0.2, 1.0).normalize();
    let brightness = 0.35 + 0.65 * best.1.dot(light).max(0.0);
    (best.0, best.2.map(|c| c * brightness))
}

/// The smallest positive `t` at which the ray meets the sphere.
fn ray_sphere(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let (a, b, c) = (dir.length_squared(), dir.dot(oc), oc.length_squared() - radius * radius);
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    (t > 0.0).then_some(t)
}

/// Where the ray first meets `o`, if before `limit`, with the outward normal there.
fn intersect(o: &Obstacle, origin: Vec3, dir: Vec3, limit: f32) -> Option<(f32, Vec3)> {
    let local = |center: Vec3, rotation: Quat| {
        let r = Mat3::from_quat(rotation);
        (r, r.transpose() * (origin - center), r.transpose() * dir)
    };
    let hit = match *o {
        Obstacle::Sphere { center, radius } => {
            ray_sphere(origin, dir, center, radius).map(|t| (t, (origin + dir * t - center) / radius))
        }
        Obstacle::Cuboid { center, half_extents, rotation } => {
            let (r, p, d) = local(center, rotation);
            ray_box(p, d, half_extents).map(|(t, n)| (t, r * n))
        }
        Obstacle::Cylinder { center, rotation, radius, half_height } => {
            let (r, p, d) = local(center, rotation);
            ray_cylinder(p, d, radius, half_height).map(|(t, n)| (t, r * n))
        }
        Obstacle::Capsule { center, rotation, radius, half_length } => {
            let (r, p, d) = local(center, rotation);
            ray_capsule(p, d, radius, half_length).map(|(t, n)| (t, r * n))
        }
        Obstacle::Sdf { ref grid, center, rotation } => {
            let (r, p, d) = local(center, rotation);
            ray_grid(grid, p, d, limit).map(|(t, n)| (t, r * n))
        }
    };
    hit.filter(|&(t, _)| t < limit)
}

/// `1 / d`, with zero components made tiny first.
fn reciprocal(d: Vec3) -> Vec3 {
    Vec3::select(d.abs().cmplt(Vec3::splat(1e-20)), Vec3::splat(1e-20), d).recip()
}

/// Slab test against the box `[-half, half]`.
fn ray_box(p: Vec3, d: Vec3, half: Vec3) -> Option<(f32, Vec3)> {
    let inv = reciprocal(d);
    let (t0, t1) = ((-half - p) * inv, (half - p) * inv);
    let (near, far) = (t0.min(t1), t0.max(t1));
    let t = near.max_element();
    if t > far.min_element() || t <= 0.0 {
        return None;
    }
    // The face entered last: the axis of the largest near distance.
    let axis = if near.x >= near.y && near.x >= near.z {
        Vec3::X
    } else if near.y >= near.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    Some((t, axis * -d.dot(axis).signum()))
}

/// A cylinder of `radius` around the z axis from `-half` to `half`: its side, then its caps.
fn ray_cylinder(p: Vec3, d: Vec3, radius: f32, half: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let (a, b, c) = (d.x * d.x + d.y * d.y, p.x * d.x + p.y * d.y, p.x * p.x + p.y * p.y - radius * radius);
    let disc = b * b - a * c;
    if a > 0.0 && disc >= 0.0 {
        let t = (-b - disc.sqrt()) / a;
        let z = p.z + d.z * t;
        if t > 0.0 && z.abs() <= half {
            best = Some((t, Vec3::new(p.x + d.x * t, p.y + d.y * t, 0.0) / radius));
        }
    }
    for side in [-1.0, 1.0] {
        if d.z != 0.0 {
            let t = (side * half - p.z) / d.z;
            let (x, y) = (p.x + d.x * t, p.y + d.y * t);
            if t > 0.0 && x * x + y * y <= radius * radius && best.is_none_or(|(b, _)| t < b) {
                best = Some((t, Vec3::Z * side));
            }
        }
    }
    best
}

/// A capsule around the z axis segment from `-half` to `half`: its side, then its end spheres.
fn ray_capsule(p: Vec3, d: Vec3, radius: f32, half: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    let (a, b, c) = (d.x * d.x + d.y * d.y, p.x * d.x + p.y * d.y, p.x * p.x + p.y * p.y - radius * radius);
    let disc = b * b - a * c;
    if a > 0.0 && disc >= 0.0 {
        let t = (-b - disc.sqrt()) / a;
        if t > 0.0 && (p.z + d.z * t).abs() <= half {
            best = Some((t, Vec3::new(p.x + d.x * t, p.y + d.y * t, 0.0) / radius));
        }
    }
    for end in [Vec3::Z * -half, Vec3::Z * half] {
        if let Some(t) = ray_sphere(p, d, end, radius)
            && best.is_none_or(|(b, _)| t < b)
        {
            best = Some((t, (p + d * t - end) / radius));
        }
    }
    best
}

/// Sphere tracing through a distance grid, from where the ray enters the grid's box (padded by
/// one voxel) until it leaves it or passes `limit`.
fn ray_grid(grid: &SdfGrid, p: Vec3, d: Vec3, limit: f32) -> Option<(f32, Vec3)> {
    let (lo, hi) = grid.bounds();
    let pad = Vec3::splat(grid.voxel);
    let inv = reciprocal(d);
    let (t0, t1) = ((lo - pad - p) * inv, (hi + pad - p) * inv);
    let (mut t, end) = (t0.min(t1).max_element().max(0.0), t0.max(t1).min_element().min(limit));
    let length = d.length();
    for _ in 0..GRID_STEPS {
        if t > end {
            return None;
        }
        let (distance, gradient) = grid_distance(grid, p + d * t);
        if distance < GRID_HIT {
            return Some((t, gradient.normalize_or_zero()));
        }
        t += distance / length;
    }
    None
}

/// Sphere tracing stops at this distance from a grid's surface, or after this many steps.
pub(crate) const GRID_HIT: f32 = 1e-3;
pub(crate) const GRID_STEPS: u32 = 128;
