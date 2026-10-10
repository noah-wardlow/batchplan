//! Signed distance grids: obstacles of any shape, built from meshes, point clouds or depth images.
//!
//! A grid holds distances at regular points as half floats, and both devices interpolate them the
//! same way (`grid_distance` here and in kernels.wgsl). The builders make grids conservative. Each
//! point holds at most the true signed distance of the geometry (for point clouds and depth
//! images, of the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear
//! interpolation can overestimate by. A built grid therefore never reads farther from an obstacle
//! than its geometry, and walls thinner than a voxel still block. The price is that obstacles grow,
//! by up to a voxel diagonal for meshes and about three voxels for points and depth images.

use anyhow::{Result, anyhow, ensure};
use glam::{UVec3, Vec3};
use half::f16;
use parry3d::query::PointQuery;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::description::TriMesh;
use crate::types::Pose;

/// Grids with more points are refused; 2^26 half floats take 128 MiB.
const MAX_POINTS: u64 = 1 << 26;

/// How builders lay out a grid.
#[derive(Clone, Copy, Debug)]
pub struct SdfOptions {
    /// Spacing of the grid points (meters).
    pub voxel: f32,
    /// How far the grid reaches beyond the geometry (meters). Beyond the grid, distances are
    /// extrapolated (see [`SdfGrid`]); keep this at least the collision margin plus the robot's
    /// largest sphere radius.
    pub padding: f32,
}

impl Default for SdfOptions {
    fn default() -> Self {
        Self { voxel: 0.01, padding: 0.15 }
    }
}

/// Pinhole intrinsics of a depth camera, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// What [`SdfGrid::from_depth`] assumes about space hidden behind observed surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Occlusion {
    /// Only observed surfaces are obstacles.
    Free,
    /// Everything behind an observed surface, as seen from the camera, is solid.
    Occupied,
}

/// Signed distances (meters, negative inside) on a regular grid in its own frame: `dims` points
/// per axis, `voxel` apart, starting at `origin`, stored as half floats with x varying fastest.
///
/// Between points, distances are interpolated trilinearly. Outside the grid's box, the distance
/// is the value at the nearest point of the box plus the distance to that point. Collision checks
/// stay exact there as long as the box reaches at least `margin + largest sphere radius` beyond
/// every surface it holds ([`SdfOptions::padding`]): both the grid and the true distance then
/// report clearance beyond the margin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SdfGrid {
    pub(crate) dims: [u32; 3],
    pub(crate) voxel: f32,
    pub(crate) origin: Vec3,
    pub(crate) values: Vec<u16>,
}

impl SdfGrid {
    /// A grid holding `distances` at `dims` points (`dims[0] * dims[1] * dims[2]` values, x
    /// fastest), as given: no conservative offset is applied.
    pub fn new(dims: [u32; 3], voxel: f32, origin: Vec3, distances: &[f32]) -> Result<Self> {
        ensure!(dims.iter().all(|&d| d >= 2), "a grid needs at least 2 points per axis, got {dims:?}");
        ensure!(voxel.is_finite() && voxel > 0.0, "the voxel size must be positive, got {voxel}");
        ensure!(origin.is_finite(), "the grid origin must be finite");
        let count = dims.iter().map(|&d| u64::from(d)).product::<u64>();
        ensure!(count <= MAX_POINTS, "{dims:?} is {count} points, more than {MAX_POINTS}; use larger voxels");
        ensure!(distances.len() as u64 == count, "{dims:?} needs {count} distances, got {}", distances.len());
        let values = distances
            .iter()
            .map(|&d| {
                let h = f16::from_f32(d);
                ensure!(h.is_finite(), "distance {d} does not fit a half float");
                // GPUs may flush subnormal halves to zero; storing them as zero keeps devices equal.
                Ok(if h.is_normal() { h.to_bits() } else { 0 })
            })
            .collect::<Result<_>>()?;
        Ok(Self { dims, voxel, origin, values })
    }

    /// The distance field of a closed triangle mesh, in the mesh's frame. Triangles may face in
    /// or out, consistently; meshes that are open or not manifold have no inside and are refused.
    pub fn from_mesh(vertices: &[Vec3], triangles: &[[u32; 3]], o: &SdfOptions) -> Result<Self> {
        ensure!(!triangles.is_empty(), "the mesh has no triangles");
        ensure!(
            triangles.iter().flatten().all(|&i| (i as usize) < vertices.len()),
            "a triangle refers to a vertex past the {} given",
            vertices.len()
        );
        ensure!(vertices.iter().all(|v| v.is_finite()), "mesh vertices must be finite");
        let mut mesh = TriMesh { vertices: vertices.to_vec(), triangles: triangles.to_vec() };
        mesh.orient_outward();
        let flags = TriMeshFlags::HALF_EDGE_TOPOLOGY
            | TriMeshFlags::ORIENTED
            | TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
        // `with_flags` drops topology errors, so the flags are set separately.
        let mut solid = ParryMesh::new(mesh.vertices.clone(), mesh.triangles)?;
        solid.set_flags(flags).map_err(|e| anyhow!("the mesh is not a manifold surface: {e}"))?;
        let topology = solid.topology().expect("set_flags computed the topology");
        let open = topology.half_edges.iter().filter(|h| h.twin == u32::MAX).count();
        ensure!(open == 0, "the mesh is not closed ({open} edges border a single triangle), so it has no inside");
        let (lo, hi) = bounds(&mesh.vertices);
        let (dims, origin) = layout(lo, hi, o)?;
        let distances = (0..dims.element_product())
            .into_par_iter()
            .map(|n| {
                let p = point(dims, origin, o.voxel, n);
                let proj = solid.project_local_point(p, false);
                let d = (p - proj.point).length();
                if proj.is_inside { -d } else { d }
            })
            .collect();
        Self::conservative(dims, o.voxel, origin, distances)
    }

    /// The distance field of surface points, such as a point cloud. Voxels holding a point are
    /// solid; the inside of a closed surface reads as free beyond its shell of voxels.
    pub fn from_points(points: &[Vec3], o: &SdfOptions) -> Result<Self> {
        ensure!(!points.is_empty(), "no points");
        ensure!(points.iter().all(|p| p.is_finite()), "points must be finite");
        let (lo, hi) = bounds(points);
        let (dims, origin) = layout(lo, hi, o)?;
        let mut occupied = vec![false; dims.element_product() as usize];
        for &p in points {
            occupied[nearest(dims, origin, o.voxel, p)] = true;
        }
        Self::from_occupancy(dims, origin, o.voxel, &occupied)
    }

    /// The distance field seen in one depth image: `depth` holds z distances in meters, row by
    /// row, `width` pixels per row, with zero, negative or non-finite values for pixels without a
    /// reading. `camera` places the camera (x right, y down, z forward, as in OpenCV) in the frame
    /// the grid is built in. The grid spans the observed points plus the padding.
    ///
    /// Unobserved space follows one rule: space outside the image and along pixels without a
    /// reading is free, and space behind observed surfaces is what `behind` says. Each voxel is
    /// classified by projecting its center into the image, so coarse sensors such as 8x8
    /// time-of-flight arrays give solid surfaces, not scattered points.
    pub fn from_depth(
        depth: &[f32],
        width: usize,
        intrinsics: Intrinsics,
        camera: Pose,
        behind: Occlusion,
        o: &SdfOptions,
    ) -> Result<Self> {
        ensure!(
            width > 0 && depth.len().is_multiple_of(width),
            "{} depth values do not form rows of {width}",
            depth.len()
        );
        let Intrinsics { fx, fy, cx, cy } = intrinsics;
        ensure!(fx > 0.0 && fy > 0.0 && cx.is_finite() && cy.is_finite(), "invalid intrinsics {intrinsics:?}");
        let height = depth.len() / width;
        let reading = |u: usize, v: usize| Some(depth[v * width + u]).filter(|z| z.is_finite() && *z > 0.0);
        let to_world = |u: f32, v: f32, z: f32| camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z);
        let points: Vec<Vec3> = (0..height)
            .flat_map(|v| (0..width).filter_map(move |u| Some((u, v, reading(u, v)?))))
            .map(|(u, v, z)| to_world(u as f32, v as f32, z) + camera.position)
            .collect();
        ensure!(!points.is_empty(), "the depth image has no valid pixels");
        let (lo, hi) = bounds(&points);
        let (dims, origin) = layout(lo, hi, o)?;
        let to_camera = camera.rotation.inverse();
        let half = 0.5 * o.voxel;
        let mut occupied: Vec<bool> = (0..dims.element_product())
            .into_par_iter()
            .map(|n| {
                let c = to_camera * (point(dims, origin, o.voxel, n) - camera.position);
                if c.z <= 0.0 {
                    return false;
                }
                let (u, v) = ((fx * c.x / c.z + cx + 0.5).floor(), (fy * c.y / c.z + cy + 0.5).floor());
                if u < 0.0 || v < 0.0 || u >= width as f32 || v >= height as f32 {
                    return false;
                }
                match reading(u as usize, v as usize) {
                    Some(z) if c.z > z + half => behind == Occlusion::Occupied,
                    Some(z) => c.z >= z - half,
                    None => false,
                }
            })
            .collect();
        // Dense images resolve surfaces finer than voxels project; their points are solid too.
        for &p in &points {
            occupied[nearest(dims, origin, o.voxel, p)] = true;
        }
        Self::from_occupancy(dims, origin, o.voxel, &occupied)
    }

    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    pub fn voxel(&self) -> f32 {
        self.voxel
    }

    /// Lower and upper corners of the grid's box, in its frame.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.origin, self.origin + (UVec3::from(self.dims) - 1).as_vec3() * self.voxel)
    }

    /// Signed distance at `p` (in the grid's frame) and its gradient.
    pub fn distance(&self, p: Vec3) -> (f32, Vec3) {
        grid_distance(self, p)
    }

    #[inline]
    fn value(&self, n: u32) -> f32 {
        f16::from_bits(self.values[n as usize]).to_f32()
    }

    /// Exact distances lowered by half a voxel diagonal (see the module documentation).
    fn conservative(dims: UVec3, voxel: f32, origin: Vec3, mut distances: Vec<f32>) -> Result<Self> {
        let shift = voxel * 3f32.sqrt() / 2.0;
        distances.iter_mut().for_each(|d| *d -= shift);
        Self::new(dims.into(), voxel, origin, &distances)
    }

    /// Signed distances to the occupied voxels as solid cubes, from their centers' distances.
    /// A free point is at least `d - sqrt(3) / 2` voxels from the nearest cube when its center is
    /// `d` away; an occupied point is at most `d - 1 / 2` deep when the nearest free center is `d`
    /// away. Both bounds keep each value at or below the true distance.
    fn from_occupancy(dims: UVec3, origin: Vec3, voxel: f32, occupied: &[bool]) -> Result<Self> {
        ensure!(!occupied.iter().all(|&o| o), "every voxel is occupied; increase the padding");
        let dims = dims.to_array().map(|d| d as usize);
        let (to_solid, to_free) = (squared_edt(occupied, true, dims), squared_edt(occupied, false, dims));
        let half_diagonal = 3f64.sqrt() / 2.0;
        let distances = (0..occupied.len())
            .map(|i| {
                let d = if occupied[i] { 0.5 - to_free[i].sqrt() } else { to_solid[i].sqrt() - half_diagonal };
                voxel * d as f32
            })
            .collect();
        Self::conservative(UVec3::from(dims.map(|d| d as u32)), voxel, origin, distances)
    }
}

/// Must stay in sync with `grid_distance` in kernels.wgsl.
#[inline]
pub(crate) fn grid_distance(g: &SdfGrid, p: Vec3) -> (f32, Vec3) {
    let top = (UVec3::from(g.dims) - 1).as_vec3();
    let x = (p - g.origin) / g.voxel;
    let c = x.clamp(Vec3::ZERO, top);
    let i = c.floor().min(top - 1.0);
    let f = c - i;
    let (sy, sz) = (g.dims[0], g.dims[0] * g.dims[1]);
    let n = i.x as u32 + sy * i.y as u32 + sz * i.z as u32;
    let v = |offset: u32| g.value(n + offset);
    let (v000, v100, v010, v110) = (v(0), v(1), v(sy), v(sy + 1));
    let (v001, v101, v011, v111) = (v(sz), v(sz + 1), v(sz + sy), v(sz + sy + 1));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let (x00, x10) = (lerp(v000, v100, f.x), lerp(v010, v110, f.x));
    let (x01, x11) = (lerp(v001, v101, f.x), lerp(v011, v111, f.x));
    let (y0, y1) = (lerp(x00, x10, f.y), lerp(x01, x11, f.y));
    let dx = lerp(lerp(v100 - v000, v110 - v010, f.y), lerp(v101 - v001, v111 - v011, f.y), f.z);
    let inside = Vec3::new(dx, lerp(x10 - x00, x11 - x01, f.z), y1 - y0) / g.voxel;
    // Outside the box, the distance grows with the distance to the box along the clamped axes.
    let outside = (x - c) * g.voxel;
    let away = outside.length();
    let grad = Vec3::select(x.cmpne(c), outside / away.max(1e-30), inside);
    (lerp(y0, y1, f.z) + away, grad)
}

fn bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    points.iter().fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)))
}

/// Grid dimensions and origin covering `lo..=hi` plus the padding.
fn layout(lo: Vec3, hi: Vec3, o: &SdfOptions) -> Result<(UVec3, Vec3)> {
    ensure!(o.voxel.is_finite() && o.voxel > 0.0, "the voxel size must be positive, got {}", o.voxel);
    ensure!(o.padding.is_finite() && o.padding >= 0.0, "the padding must not be negative, got {}", o.padding);
    let origin = lo - o.padding;
    let dims = ((hi + o.padding - origin) / o.voxel).ceil().max(Vec3::ONE) + 1.0;
    let points = dims.as_dvec3().element_product();
    ensure!(points <= MAX_POINTS as f64, "a grid of {} m voxels over {} m is too large", o.voxel, hi - lo);
    Ok((dims.as_uvec3(), origin))
}

fn point(dims: UVec3, origin: Vec3, voxel: f32, n: u32) -> Vec3 {
    let (x, y, z) = (n % dims.x, n / dims.x % dims.y, n / (dims.x * dims.y));
    origin + UVec3::new(x, y, z).as_vec3() * voxel
}

fn nearest(dims: UVec3, origin: Vec3, voxel: f32, p: Vec3) -> usize {
    let i = ((p - origin) / voxel).round().max(Vec3::ZERO).as_uvec3().min(dims - 1);
    (i.x + dims.x * (i.y + dims.y * i.z)) as usize
}

/// Squared distance, in voxels, from every voxel to the nearest one where `sites[i] == target`:
/// Felzenszwalb and Huttenlocher's separable transform, one axis after another.
fn squared_edt(sites: &[bool], target: bool, [nx, ny, nz]: [usize; 3]) -> Vec<f64> {
    let mut d: Vec<f64> = sites.iter().map(|&s| if s == target { 0.0 } else { f64::INFINITY }).collect();
    d.par_chunks_mut(nx * ny).for_each(|slab| {
        let mut line = Line::default();
        for row in slab.chunks_mut(nx) {
            line.transform(row.iter().copied());
            row.copy_from_slice(&line.out);
        }
        for x in 0..nx {
            line.transform((0..ny).map(|y| slab[x + nx * y]));
            (0..ny).for_each(|y| slab[x + nx * y] = line.out[y]);
        }
    });
    let columns: Vec<Vec<f64>> = (0..nx * ny)
        .into_par_iter()
        .map_init(Line::default, |line, c| {
            line.transform((0..nz).map(|z| d[c + nx * ny * z]));
            line.out.clone()
        })
        .collect();
    for (c, column) in columns.iter().enumerate() {
        (0..nz).for_each(|z| d[c + nx * ny * z] = column[z]);
    }
    d
}

/// One-dimensional squared distance transform: the lower envelope of parabolas rooted at the
/// finite input values.
#[derive(Default)]
struct Line {
    f: Vec<f64>,
    roots: Vec<usize>,
    starts: Vec<f64>,
    out: Vec<f64>,
}

impl Line {
    fn transform(&mut self, f: impl Iterator<Item = f64>) {
        self.f.clear();
        self.f.extend(f);
        let (f, roots, starts) = (&self.f, &mut self.roots, &mut self.starts);
        roots.clear();
        starts.clear();
        let meet = |p: usize, q: usize| ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2 * (q - p)) as f64;
        for q in (0..f.len()).filter(|&q| f[q].is_finite()) {
            let mut s = f64::NEG_INFINITY;
            while let Some(&p) = roots.last() {
                s = meet(p, q);
                if s > *starts.last().expect("one start per root") {
                    break;
                }
                roots.pop();
                starts.pop();
                s = f64::NEG_INFINITY;
            }
            roots.push(q);
            starts.push(s);
        }
        self.out.clear();
        if roots.is_empty() {
            self.out.resize(f.len(), f64::INFINITY);
            return;
        }
        let mut k = 0;
        for q in 0..f.len() {
            while k + 1 < roots.len() && starts[k + 1] < q as f64 {
                k += 1;
            }
            let d = q as f64 - roots[k] as f64;
            self.out.push(d * d + f[roots[k]]);
        }
    }
}
