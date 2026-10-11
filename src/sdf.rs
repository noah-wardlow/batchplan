//! Signed distance grids: obstacles of any shape, built from meshes, point clouds or depth images.
//!
//! A grid holds distances at regular points as half floats, and both devices interpolate them the
//! same way (`grid_distance` here and in kernels.wgsl). The builders make grids conservative. Each
//! point holds at most the true signed distance of the geometry (for point clouds and depth
//! images, of the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear
//! interpolation can overestimate by. A built grid therefore never reads farther from an obstacle
//! than its geometry, and walls thinner than a voxel still block. The price is that obstacles grow,
//! by up to a voxel diagonal for meshes and about three voxels for points and depth images.

use crate::error::{Error, Result, ensure_input, input};
use glam::{UVec3, Vec3};
use half::f16;
use parry3d::query::PointQuery;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::description::TriMesh;
use crate::robot::Robot;
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

/// One depth image: z distances in meters, row by row, `width` pixels per row, with zero, negative
/// or non-finite values for pixels without a reading. `camera` places the camera (x right, y down,
/// z forward, as in OpenCV) in the frame grids and maps are built in.
#[derive(Clone, Copy, Debug)]
pub struct DepthImage<'a> {
    pub depth: &'a [f32],
    pub width: usize,
    pub intrinsics: Intrinsics,
    pub camera: Pose,
}

impl DepthImage<'_> {
    fn check(&self) -> Result<()> {
        ensure_input!(
            self.width > 0 && self.depth.len().is_multiple_of(self.width),
            "{} depth values do not form rows of {}",
            self.depth.len(),
            self.width
        );
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        ensure_input!(
            fx > 0.0 && fy > 0.0 && cx.is_finite() && cy.is_finite(),
            "invalid intrinsics {:?}",
            self.intrinsics
        );
        Ok(())
    }

    fn height(&self) -> usize {
        self.depth.len() / self.width
    }

    fn reading(&self, i: usize) -> Option<f32> {
        Some(self.depth[i]).filter(|z| z.is_finite() && *z > 0.0)
    }

    /// The pixel `p` (world frame) projects to, and its depth; `None` behind or beside the camera.
    fn project(&self, p: Vec3) -> Option<(usize, f32)> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let c = self.camera.rotation.inverse() * (p - self.camera.position);
        if c.z <= 0.0 {
            return None;
        }
        let (u, v) = ((fx * c.x / c.z + cx + 0.5).floor(), (fy * c.y / c.z + cy + 0.5).floor());
        (u >= 0.0 && v >= 0.0 && u < self.width as f32 && v < self.height() as f32)
            .then(|| (v as usize * self.width + u as usize, c.z))
    }

    /// Each reading as a world point.
    fn points(&self) -> impl Iterator<Item = (usize, Vec3)> + '_ {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        (0..self.depth.len()).filter_map(move |i| {
            let z = self.reading(i)?;
            let (u, v) = ((i % self.width) as f32, (i / self.width) as f32);
            Some((i, self.camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z) + self.camera.position))
        })
    }

    /// Each pixel's depth to `robot` at configuration `q`, its collision spheres grown by
    /// `padding`; infinite where the ray misses the robot. Pass it to [`SdfGrid::from_depth`] or
    /// [`OccupancyMap::integrate`] so readings of the robot itself do not become obstacles.
    pub fn robot_depth(&self, robot: &Robot, q: &[f32], padding: f32) -> Result<Vec<f32>> {
        self.check()?;
        ensure_input!(q.len() == robot.dof(), "q has {} values, the robot {} joints", q.len(), robot.dof());
        ensure_input!(padding.is_finite() && padding >= 0.0, "the padding must not be negative, got {padding}");
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let (width, height) = (self.width, self.height());
        let fk = robot.fk(q);
        let to_camera = self.camera.rotation.inverse();
        let mut out = vec![f32::INFINITY; self.depth.len()];
        for sphere in &robot.spheres {
            let center = to_camera * (fk.rot[sphere.link] * sphere.center + fk.pos[sphere.link] - self.camera.position);
            let r = sphere.radius + padding;
            if center.z + r <= 0.0 {
                continue;
            }
            // The pixels the sphere can cover: the projections of its bounding box's sides at its
            // near and far depths. The whole image when the camera is inside or next to it.
            let (near, far) = (center.z - r, center.z + r);
            let span = |c: f32, f: f32, k: f32, n: usize| {
                if near <= 1e-3 {
                    return (0, n);
                }
                let ends = [(c - r) / near, (c - r) / far, (c + r) / near, (c + r) / far].map(|x| f * x + k);
                let lo = ends.iter().fold(f32::INFINITY, |m, &e| m.min(e)).floor().max(0.0);
                let hi = ends.iter().fold(f32::NEG_INFINITY, |m, &e| m.max(e)).ceil() + 1.0;
                (lo.min(n as f32) as usize, hi.clamp(0.0, n as f32) as usize)
            };
            let ((u0, u1), (v0, v1)) = (span(center.x, fx, cx, width), span(center.y, fy, cy, height));
            for v in v0..v1 {
                for u in u0..u1 {
                    // Points t * ray along the pixel; their depth is t.
                    let ray = Vec3::new((u as f32 - cx) / fx, (v as f32 - cy) / fy, 1.0);
                    let (a, b, c) = (ray.length_squared(), ray.dot(center), center.length_squared() - r * r);
                    let disc = b * b - a * c;
                    if disc < 0.0 {
                        continue;
                    }
                    let t = ((b - disc.sqrt()) / a).max(0.0);
                    let o = &mut out[v * width + u];
                    *o = o.min(t);
                }
            }
        }
        Ok(out)
    }
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
#[serde(try_from = "GridParts")]
pub struct SdfGrid {
    pub(crate) dims: [u32; 3],
    pub(crate) voxel: f32,
    pub(crate) origin: Vec3,
    pub(crate) values: Vec<u16>,
}

/// A grid as serialized, checked like [`SdfGrid::new`] before it becomes one.
#[derive(Deserialize)]
struct GridParts {
    dims: [u32; 3],
    voxel: f32,
    origin: Vec3,
    values: Vec<u16>,
}

impl TryFrom<GridParts> for SdfGrid {
    type Error = Error;

    fn try_from(p: GridParts) -> Result<Self> {
        let grid = SdfGrid { dims: p.dims, voxel: p.voxel, origin: p.origin, values: p.values };
        grid.check()?;
        Ok(grid)
    }
}

impl SdfGrid {
    /// A grid holding `distances` at `dims` points (`dims[0] * dims[1] * dims[2]` values, x
    /// fastest), rounded to the nearest half float: no conservative offset is applied.
    pub fn new(dims: [u32; 3], voxel: f32, origin: Vec3, distances: &[f32]) -> Result<Self> {
        let values = distances.iter().map(|&d| flush(f16::from_f32(d)).to_bits()).collect();
        let grid = Self { dims, voxel, origin, values };
        grid.check()?;
        Ok(grid)
    }

    /// Why this grid cannot be sampled, if it cannot: the rules of [`SdfGrid::new`].
    pub(crate) fn check(&self) -> Result<()> {
        let (dims, voxel) = (self.dims, self.voxel);
        ensure_input!(dims.iter().all(|&d| d >= 2), "a grid needs at least 2 points per axis, got {dims:?}");
        ensure_input!(voxel.is_finite() && voxel > 0.0, "the voxel size must be positive, got {voxel}");
        ensure_input!(self.origin.is_finite(), "the grid origin must be finite");
        let count = dims.iter().map(|&d| u64::from(d)).product::<u64>();
        ensure_input!(count <= MAX_POINTS, "{dims:?} is {count} points, more than {MAX_POINTS}; use larger voxels");
        ensure_input!(self.values.len() as u64 == count, "{dims:?} needs {count} distances, got {}", self.values.len());
        let bad = self.values.iter().map(|&v| f16::from_bits(v)).find(|h| !h.is_finite() || subnormal(*h));
        ensure_input!(bad.is_none(), "distance {bad:?} is not finite or is subnormal");
        Ok(())
    }

    /// The distance field of a closed triangle mesh, in the mesh's frame. Triangles may face in
    /// or out, consistently; meshes that are open or not manifold have no inside and are refused.
    pub fn from_mesh(vertices: &[Vec3], triangles: &[[u32; 3]], o: &SdfOptions) -> Result<Self> {
        ensure_input!(!triangles.is_empty(), "the mesh has no triangles");
        ensure_input!(
            triangles.iter().flatten().all(|&i| (i as usize) < vertices.len()),
            "a triangle refers to a vertex past the {} given",
            vertices.len()
        );
        ensure_input!(vertices.iter().all(|v| v.is_finite()), "mesh vertices must be finite");
        let mut mesh = TriMesh { vertices: vertices.to_vec(), triangles: triangles.to_vec() };
        mesh.orient_outward();
        let flags = TriMeshFlags::HALF_EDGE_TOPOLOGY
            | TriMeshFlags::ORIENTED
            | TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
        // `with_flags` drops topology errors, so the flags are set separately.
        let mut solid = ParryMesh::new(mesh.vertices.clone(), mesh.triangles).map_err(|e| input!("{e}"))?;
        solid.set_flags(flags).map_err(|e| input!("the mesh is not a manifold surface: {e}"))?;
        let topology = solid.topology().expect("set_flags computed the topology");
        let open = topology.half_edges.iter().filter(|h| h.twin == u32::MAX).count();
        ensure_input!(open == 0, "the mesh is not closed ({open} edges border a single triangle), so it has no inside");
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
        ensure_input!(!points.is_empty(), "no points");
        ensure_input!(points.iter().all(|p| p.is_finite()), "points must be finite");
        let (lo, hi) = bounds(points);
        let (dims, origin) = layout(lo, hi, o)?;
        let mut occupied = vec![false; dims.element_product() as usize];
        for &p in points {
            occupied[nearest(dims, origin, o.voxel, p)] = true;
        }
        Self::from_occupancy(dims, origin, o.voxel, &occupied)
    }

    /// The distance field seen in one depth image. The grid spans the observed points plus the
    /// padding.
    ///
    /// Unobserved space follows one rule: space outside the image and along pixels without a
    /// reading is free, and space behind observed surfaces is what `behind` says. Each voxel is
    /// classified by projecting its center into the image, so coarse sensors such as 8x8
    /// time-of-flight arrays give solid surfaces, not scattered points. With `robot`, each pixel's
    /// depth to the robot ([`DepthImage::robot_depth`]), readings on or behind the robot are not
    /// obstacles: space in front of it is free and space behind it is what `behind` says.
    pub fn from_depth(image: &DepthImage, robot: Option<&[f32]>, behind: Occlusion, o: &SdfOptions) -> Result<Self> {
        image.check()?;
        if let Some(r) = robot {
            ensure_input!(r.len() == image.depth.len(), "{} robot depths for {} pixels", r.len(), image.depth.len());
        }
        // A reading at or beyond the robot (within a voxel) sees the robot.
        let robot_at = |i: usize| robot.map_or(f32::INFINITY, |r| r[i]);
        let on_robot = |i: usize, z: f32| z >= robot_at(i) - o.voxel;
        let points: Vec<Vec3> = image.points().filter(|&(i, _)| !on_robot(i, image.depth[i])).map(|(_, p)| p).collect();
        ensure_input!(!points.is_empty(), "the depth image has no valid pixels off the robot");
        let (lo, hi) = bounds(&points);
        let (dims, origin) = layout(lo, hi, o)?;
        let half = 0.5 * o.voxel;
        let mut occupied: Vec<bool> = (0..dims.element_product())
            .into_par_iter()
            .map(|n| {
                let Some((i, z_voxel)) = image.project(point(dims, origin, o.voxel, n)) else { return false };
                match image.reading(i) {
                    Some(z) if on_robot(i, z) => z_voxel >= robot_at(i) - half && behind == Occlusion::Occupied,
                    Some(z) if z_voxel > z + half => behind == Occlusion::Occupied,
                    Some(z) => z_voxel >= z - half,
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

    /// Exact distances lowered by half a voxel diagonal (see the module documentation), each
    /// rounded down to a half float so the stored value never exceeds it.
    fn conservative(dims: UVec3, voxel: f32, origin: Vec3, distances: Vec<f32>) -> Result<Self> {
        let shift = voxel * 3f32.sqrt() / 2.0;
        let values = distances.iter().map(|&d| at_most(d - shift).to_bits()).collect();
        let grid = Self { dims: dims.into(), voxel, origin, values };
        grid.check()?;
        Ok(grid)
    }

    /// Signed distances to the occupied voxels as solid cubes, from their centers' distances.
    /// A free point is at least `d - sqrt(3) / 2` voxels from the nearest cube when its center is
    /// `d` away; an occupied point is at most `d - 1 / 2` deep when the nearest free center is `d`
    /// away. Both bounds keep each value at or below the true distance.
    fn from_occupancy(dims: UVec3, origin: Vec3, voxel: f32, occupied: &[bool]) -> Result<Self> {
        ensure_input!(!occupied.iter().all(|&o| o), "every voxel is occupied; increase the padding");
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

fn subnormal(h: f16) -> bool {
    h.classify() == std::num::FpCategory::Subnormal
}

/// GPUs may flush subnormal halves to zero; storing them as zero keeps devices equal.
fn flush(h: f16) -> f16 {
    if subnormal(h) { f16::ZERO } else { h }
}

/// The largest half float that is not subnormal and not above `d` (infinite when `d` overflows).
fn at_most(d: f32) -> f16 {
    let mut h = f16::from_f32(d);
    if h.to_f32() > d {
        h = match h.to_bits() {
            0x0000 | 0x8000 => f16::from_bits(0x8001),
            bits if h.is_sign_negative() => f16::from_bits(bits + 1),
            bits => f16::from_bits(bits - 1),
        };
    }
    match subnormal(h) {
        true if h.is_sign_negative() => -f16::MIN_POSITIVE,
        true => f16::ZERO,
        false => h,
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
    ensure_input!(o.voxel.is_finite() && o.voxel > 0.0, "the voxel size must be positive, got {}", o.voxel);
    ensure_input!(o.padding.is_finite() && o.padding >= 0.0, "the padding must not be negative, got {}", o.padding);
    let origin = lo - o.padding;
    let dims = ((hi + o.padding - origin) / o.voxel).ceil().max(Vec3::ONE) + 1.0;
    let points = dims.as_dvec3().element_product();
    ensure_input!(points <= MAX_POINTS as f64, "a grid of {} m voxels over {} m is too large", o.voxel, hi - lo);
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

/// Log-odds a hit adds, a miss adds, and their clamps: OctoMap's.
const HIT: f32 = 0.85;
const MISS: f32 = -0.4;
const CLAMP: (f32, f32) = (-2.0, 3.5);

/// Occupancy fused from depth images over time, on a fixed grid in the world frame: per voxel, the
/// log-odds that it is occupied, updated as OctoMap updates them (a hit adds 0.85, a miss -0.4,
/// clamped to [-2, 3.5]). Readings that come and go settle: something seen once and then seen
/// through three times is free again. Build distance grids from it with [`OccupancyMap::grid`].
#[derive(Clone, Debug)]
pub struct OccupancyMap {
    dims: UVec3,
    origin: Vec3,
    voxel: f32,
    log_odds: Vec<f32>,
    observed: Vec<bool>,
}

impl OccupancyMap {
    /// An unobserved map with `voxel` spacing whose points cover `lo..=hi`.
    pub fn new(lo: Vec3, hi: Vec3, voxel: f32) -> Result<Self> {
        ensure_input!(lo.is_finite() && hi.is_finite() && lo.cmplt(hi).all(), "a map needs lo < hi, got {lo} and {hi}");
        let (dims, origin) = layout(lo, hi, &SdfOptions { voxel, padding: 0.0 })?;
        let n = dims.element_product() as usize;
        Ok(Self { dims, origin, voxel, log_odds: vec![0.0; n], observed: vec![false; n] })
    }

    /// Fuses one depth image. Each voxel is projected into it: voxels in front of a reading are
    /// misses, voxels at it (within half a voxel) and those holding a reading's point are hits, and
    /// voxels behind readings, outside the image or along pixels without a reading are not updated.
    /// With `robot`, each pixel's depth to the robot ([`DepthImage::robot_depth`]), pixels that see
    /// the robot only clear the space in front of it.
    pub fn integrate(&mut self, image: &DepthImage, robot: Option<&[f32]>) -> Result<()> {
        image.check()?;
        if let Some(r) = robot {
            ensure_input!(r.len() == image.depth.len(), "{} robot depths for {} pixels", r.len(), image.depth.len());
        }
        let (dims, origin, voxel) = (self.dims, self.origin, self.voxel);
        let half = 0.5 * voxel;
        let robot_at = |i: usize| robot.map_or(f32::INFINITY, |r| r[i]);
        let on_robot = |i: usize, z: f32| z >= robot_at(i) - voxel;
        // This frame's update of each voxel: -1 a miss, 1 a hit.
        let mut update: Vec<i8> = (0..dims.element_product())
            .into_par_iter()
            .map(|n| {
                let Some((i, z_voxel)) = image.project(point(dims, origin, voxel, n)) else { return 0 };
                match image.reading(i) {
                    Some(z) if on_robot(i, z) => -i8::from(z_voxel < robot_at(i) - half),
                    Some(z) if z_voxel < z - half => -1,
                    Some(z) if z_voxel <= z + half => 1,
                    _ => 0,
                }
            })
            .collect();
        let (lo, hi) = (origin - half, origin + (dims - 1).as_vec3() * voxel + half);
        for (i, p) in image.points() {
            if !on_robot(i, image.depth[i]) && p.cmpge(lo).all() && p.cmplt(hi).all() {
                update[nearest(dims, origin, voxel, p)] = 1;
            }
        }
        self.log_odds.par_iter_mut().zip(self.observed.par_iter_mut()).zip(update).for_each(|((l, seen), u)| {
            if u != 0 {
                *seen = true;
                *l = (*l + if u > 0 { HIT } else { MISS }).clamp(CLAMP.0, CLAMP.1);
            }
        });
        Ok(())
    }

    /// The probability that the voxel holding `p` is occupied; `None` if it was never observed or
    /// lies outside the map.
    pub fn occupancy(&self, p: Vec3) -> Option<f32> {
        let half = 0.5 * self.voxel;
        let (lo, hi) = (self.origin - half, self.origin + (self.dims - 1).as_vec3() * self.voxel + half);
        if !(p.cmpge(lo).all() && p.cmplt(hi).all()) {
            return None;
        }
        let i = nearest(self.dims, self.origin, self.voxel, p);
        self.observed[i].then(|| 1.0 / (1.0 + (-self.log_odds[i]).exp()))
    }

    /// The distance grid of the map: voxels more likely occupied than not are solid, and voxels
    /// never observed are solid or free as `unknown` says. Errors if nothing is solid.
    pub fn grid(&self, unknown: Occlusion) -> Result<SdfGrid> {
        let occupied: Vec<bool> = self
            .log_odds
            .iter()
            .zip(&self.observed)
            .map(|(&l, &seen)| if seen { l > 0.0 } else { unknown == Occlusion::Occupied })
            .collect();
        ensure_input!(occupied.iter().any(|&o| o), "the map holds nothing occupied");
        SdfGrid::from_occupancy(self.dims, self.origin, self.voxel, &occupied)
    }
}
