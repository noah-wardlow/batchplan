//! Format-agnostic robot descriptions. Every loader produces a [`RobotDescription`]; kinematics,
//! collision spheres and self-collision analysis are built from it, so they never depend on the
//! file format a robot came from.

use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::Vec3;

use crate::robot::Transform;

#[derive(Clone, Debug)]
pub(crate) struct RobotDescription {
    pub(crate) name: String,
    pub(crate) links: Vec<LinkDesc>,
    pub(crate) joints: Vec<JointDesc>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LinkDesc {
    pub(crate) name: String,
    pub(crate) collision: Vec<Shape>,
    pub(crate) visual: Vec<Shape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointType {
    Fixed,
    Revolute,
    /// Revolute without position limits.
    Continuous,
    Prismatic,
}

#[derive(Clone, Debug)]
pub(crate) struct JointDesc {
    pub(crate) name: String,
    pub(crate) kind: JointType,
    pub(crate) parent: String,
    pub(crate) child: String,
    /// Joint frame relative to the parent link frame; the child link frame at zero motion.
    pub(crate) origin: Transform,
    /// Unit motion axis in the joint frame.
    pub(crate) axis: Vec3,
    pub(crate) lower: f32,
    pub(crate) upper: f32,
    /// Infinite when the description gives none.
    pub(crate) max_velocity: f32,
    pub(crate) mimic: Option<Mimic>,
}

/// `value = multiplier * value(joint) + offset`.
#[derive(Clone, Debug)]
pub(crate) struct Mimic {
    pub(crate) joint: String,
    pub(crate) multiplier: f32,
    pub(crate) offset: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct Shape {
    /// Shape frame relative to its link frame.
    pub(crate) origin: Transform,
    pub(crate) geometry: Geometry,
}

/// Primitive sizes are half extents; round primitives are aligned with their local z axis.
#[derive(Clone, Debug)]
pub(crate) enum Geometry {
    Box { half: Vec3 },
    Sphere { radius: f32 },
    Cylinder { radius: f32, half_length: f32 },
    Capsule { radius: f32, half_length: f32 },
    Mesh { path: PathBuf, scale: Vec3 },
}

/// A triangle mesh with outward-facing (counterclockwise) triangles.
#[derive(Clone, Debug, Default)]
pub(crate) struct TriMesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    fn append(&mut self, other: TriMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.triangles.extend(other.triangles.iter().map(|t| t.map(|i| i + base)));
    }

    pub(crate) fn corners(&self, t: [u32; 3]) -> [Vec3; 3] {
        t.map(|i| self.vertices[i as usize])
    }

    /// Six times the enclosed volume; negative when the triangles face inward.
    fn signed_volume6(&self) -> f32 {
        self.triangles
            .iter()
            .map(|&t| {
                let [a, b, c] = self.corners(t);
                a.dot(b.cross(c))
            })
            .sum()
    }
}

pub(crate) fn triangle_area([a, b, c]: [Vec3; 3]) -> f32 {
    0.5 * (b - a).cross(c - a).length()
}

impl Shape {
    /// The shape as an outward-facing triangle mesh in its link's frame.
    pub(crate) fn mesh(&self) -> Result<TriMesh> {
        let mut mesh = match &self.geometry {
            Geometry::Box { half } => box_mesh(*half),
            Geometry::Sphere { radius } => {
                let profile: Vec<(f32, f32)> = (0..=12).map(|k| polar(*radius, PI * k as f32 / 12.0, 0.0)).collect();
                lathe(&profile)
            }
            Geometry::Cylinder { radius, half_length } => {
                lathe(&[(0.0, -half_length), (*radius, -half_length), (*radius, *half_length), (0.0, *half_length)])
            }
            Geometry::Capsule { radius, half_length } => {
                let mut profile: Vec<(f32, f32)> =
                    (0..=6).map(|k| polar(*radius, PI * k as f32 / 12.0, -half_length)).collect();
                profile.extend((6..=12).map(|k| polar(*radius, PI * k as f32 / 12.0, *half_length)));
                lathe(&profile)
            }
            Geometry::Mesh { path, scale } => load_mesh(path, *scale)?,
        };
        if mesh.signed_volume6() < 0.0 {
            mesh.triangles.iter_mut().for_each(|t| t.swap(1, 2));
        }
        for v in &mut mesh.vertices {
            *v = self.origin.rot * *v + self.origin.trans;
        }
        Ok(mesh)
    }
}

/// Point of a profile running from the bottom pole (`angle` 0) to the top pole (`angle` pi),
/// as (radius, z).
fn polar(radius: f32, angle: f32, z_offset: f32) -> (f32, f32) {
    let s = angle.sin();
    (if s.abs() < 1e-6 { 0.0 } else { radius * s }, -radius * angle.cos() + z_offset)
}

/// Revolves a (radius, z) profile from bottom to top about the z axis. Profile points with zero
/// radius become single pole vertices.
fn lathe(profile: &[(f32, f32)]) -> TriMesh {
    const SECTORS: u32 = 24;
    let mut mesh = TriMesh::default();
    let rings: Vec<Vec<u32>> = profile
        .iter()
        .map(|&(r, z)| {
            let first = mesh.vertices.len() as u32;
            if r <= 0.0 {
                mesh.vertices.push(Vec3::new(0.0, 0.0, z));
                return vec![first; SECTORS as usize];
            }
            mesh.vertices.extend((0..SECTORS).map(|j| {
                let a = TAU * j as f32 / SECTORS as f32;
                Vec3::new(r * a.cos(), r * a.sin(), z)
            }));
            (first..first + SECTORS).collect()
        })
        .collect();
    for pair in rings.windows(2) {
        let (lo, hi) = (&pair[0], &pair[1]);
        for j in 0..SECTORS as usize {
            let k = (j + 1) % SECTORS as usize;
            let (a, b, c, d) = (lo[j], lo[k], hi[k], hi[j]);
            if a != b {
                mesh.triangles.push([a, b, c]);
            }
            if c != d {
                mesh.triangles.push([a, c, d]);
            }
        }
    }
    mesh
}

fn box_mesh(half: Vec3) -> TriMesh {
    // Vertex i has coordinate +half where bit (x, y, z) = (1, 2, 4) of i is set.
    let vertices =
        (0..8).map(|i| Vec3::new(sign(i & 1) * half.x, sign(i & 2) * half.y, sign(i & 4) * half.z)).collect();
    let triangles = vec![
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5], // -x, +x
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3], // -y, +y
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6], // -z, +z
    ];
    TriMesh { vertices, triangles }
}

fn sign(bit: u32) -> f32 {
    if bit != 0 { 1.0 } else { -1.0 }
}

fn load_mesh(path: &Path, scale: Vec3) -> Result<TriMesh> {
    let scene =
        mesh_loader::Loader::default().load(path).with_context(|| format!("loading mesh {}", path.display()))?;
    let mut mesh = TriMesh::default();
    for m in scene.meshes {
        mesh.append(TriMesh {
            vertices: m.vertices.iter().map(|&v| Vec3::from(v) * scale).collect(),
            triangles: m.faces,
        });
    }
    if mesh.triangles.is_empty() {
        bail!("mesh {} has no triangles", path.display());
    }
    Ok(mesh)
}

/// Loads a robot description, choosing the format by file extension.
pub(crate) fn load_robot(path: &Path, package_dirs: &[PathBuf]) -> Result<RobotDescription> {
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("urdf") => crate::urdf::load(path, package_dirs),
        _ => bail!("{}: unsupported robot description format (expected .urdf)", path.display()),
    }
}
