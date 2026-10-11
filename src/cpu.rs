//! CPU reference backend (rayon over items, on a pool it owns). Every function here has a WGSL twin in
//! `kernels.wgsl`; the GPU parity tests compare the two.

// Index loops here mirror kernels.wgsl line for line, which keeps the two easy to compare.
#![allow(clippy::needless_range_loop)]

use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{Error, Result};
use glam::{Mat3, Vec3};
use rayon::prelude::*;

use crate::device::{Backend, CollisionWeights, Evaluation, Worlds};
use crate::ik::IkOptions;
use crate::robot::{Fk, JointKind, Robot};
use crate::sdf::SdfGrid;
use crate::spline;
use crate::trajopt::{LINE_SEARCH, MAX_HISTORY, PlanOptions};
use crate::types::{JointPaths, Pose};
use crate::world::{
    FAR, Obstacle, World, box_distance, capsule_distance, cylinder_distance, sdf_distance, sphere_distance,
};

pub(crate) struct CpuBackend {
    robot: Robot,
    /// All batched work runs here, never on rayon's global pool.
    pool: Arc<rayon::ThreadPool>,
}

impl CpuBackend {
    /// `threads` workers; 0 means one per core.
    pub(crate) fn new(robot: &Robot, threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("batchplan-cpu-{i}"))
            .build()
            .map_err(|e| Error::Threads(format!("starting the CPU device's threads: {e}")))?;
        Ok(Self { robot: robot.clone(), pool: Arc::new(pool) })
    }
}

/// Obstacle with its rotation matrix precomputed.
enum Prepared {
    Cuboid { rot: Mat3, center: Vec3, half: Vec3 },
    Sphere { center: Vec3, radius: f32 },
    Cylinder { rot: Mat3, center: Vec3, radius: f32, half_height: f32 },
    Capsule { rot: Mat3, center: Vec3, radius: f32, half_length: f32 },
    Sdf { rot: Mat3, center: Vec3, grid: Arc<SdfGrid> },
}

/// The obstacles of each world, prepared.
type CpuWorlds = Vec<Vec<Prepared>>;

fn prepare(worlds: &[World]) -> CpuWorlds {
    worlds
        .iter()
        .map(|s| {
            s.obstacles
                .iter()
                .map(|o| match *o {
                    Obstacle::Cuboid { center, half_extents, rotation } => {
                        Prepared::Cuboid { rot: Mat3::from_quat(rotation), center, half: half_extents }
                    }
                    Obstacle::Sphere { center, radius } => Prepared::Sphere { center, radius },
                    Obstacle::Cylinder { center, rotation, radius, half_height } => {
                        Prepared::Cylinder { rot: Mat3::from_quat(rotation), center, radius, half_height }
                    }
                    Obstacle::Capsule { center, rotation, radius, half_length } => {
                        Prepared::Capsule { rot: Mat3::from_quat(rotation), center, radius, half_length }
                    }
                    Obstacle::Sdf { ref grid, center, rotation } => {
                        Prepared::Sdf { rot: Mat3::from_quat(rotation), center, grid: grid.clone() }
                    }
                })
                .collect()
        })
        .collect()
}

struct CollisionOut {
    cost: f32,
    world_clearance: f32,
    self_clearance: f32,
}

/// A configuration's sphere centres and per-link collision wrenches (force, and moment about the
/// world origin).
struct CollisionBuffers {
    centers: Vec<Vec3>,
    force: Vec<Vec3>,
    moment: Vec<Vec3>,
}

/// Working memory for one configuration at a time, sized to the robot. The hot loops keep one per
/// thread and reuse it, so they neither allocate nor clear more than the robot needs.
struct Scratch {
    fk: Fk,
    q: Vec<f32>,
    collision: CollisionBuffers,
    /// IK: the end effector's Jacobian (6 x dof, row-major), the step and the collision gradient.
    jac: Vec<f32>,
    dq: Vec<f32>,
    grad: Vec<f32>,
}

impl Scratch {
    fn new(robot: &Robot) -> Self {
        let (n, links) = (robot.dof(), robot.links.len());
        Self {
            fk: Fk::new(links),
            q: vec![0.0; n],
            collision: CollisionBuffers {
                centers: vec![Vec3::ZERO; robot.spheres.len()],
                force: vec![Vec3::ZERO; links],
                moment: vec![Vec3::ZERO; links],
            },
            jac: vec![0.0; 6 * n],
            dq: vec![0.0; n],
            grad: vec![0.0; n],
        }
    }
}

/// Signed distance from `o` to `p` and its gradient. Mirrors `obstacle_distance` in kernels.wgsl.
#[inline]
fn obstacle_distance(o: &Prepared, p: Vec3) -> (f32, Vec3) {
    match *o {
        Prepared::Cuboid { rot, center, half } => box_distance(rot, center, half, p),
        Prepared::Sphere { center, radius } => sphere_distance(center, radius, p),
        Prepared::Cylinder { rot, center, radius, half_height } => {
            cylinder_distance(rot, center, radius, half_height, p)
        }
        Prepared::Capsule { rot, center, radius, half_length } => capsule_distance(rot, center, radius, half_length, p),
        Prepared::Sdf { rot, center, ref grid } => sdf_distance(rot, center, grid, p),
    }
}

/// Collision cost of the configuration behind `fk`; adds d(cost)/dq into `grad`.
/// Without `GRADIENT` only the cost and clearances are computed, as the WGSL `gradient` flag does.
// Inlined into every caller: once the cost-only form had two callers, LLVM stopped inlining it into
// the line search, which cost 5% of CPU planning.
#[inline(always)]
fn collision<const GRADIENT: bool>(
    robot: &Robot,
    world: &[Prepared],
    fk: &Fk,
    w: &CollisionWeights,
    grad: &mut [f32],
    buffers: &mut CollisionBuffers,
) -> CollisionOut {
    let (rot, pos) = (&fk.rot[..], &fk.pos[..]);
    let CollisionBuffers { centers: sc, force, moment } = buffers;
    for (c, sp) in sc.iter_mut().zip(&robot.spheres) {
        *c = rot[sp.link] * sp.center + pos[sp.link];
    }
    let sc = &sc[..];
    if GRADIENT {
        force.fill(Vec3::ZERO);
        moment.fill(Vec3::ZERO);
    }
    let mut touched = false;
    let (mut cost, mut wmin, mut smin) = (0.0f32, FAR, FAR);
    // An obstacle farther from a link's bounding sphere than the margin cannot add cost through the
    // link's spheres; the gap bounds their clearance from below. Distance grids are interpolated,
    // not exact, so their spheres are always checked.
    let world_gate = w.margin.max(0.0);
    for (link, &[first, count]) in robot.sphere_ranges.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let b = robot.link_bounds[link];
        let bc = rot[link] * Vec3::new(b[0], b[1], b[2]) + pos[link];
        for o in world {
            if !matches!(o, Prepared::Sdf { .. }) {
                let gap = obstacle_distance(o, bc).0 - b[3];
                if gap > world_gate {
                    wmin = wmin.min(gap);
                    continue;
                }
            }
            for s in first as usize..(first + count) as usize {
                let (dist, g) = obstacle_distance(o, sc[s]);
                let d = dist - robot.spheres[s].radius;
                wmin = wmin.min(d);
                let pen = w.margin - d;
                if pen > 0.0 {
                    cost += w.world * pen * pen;
                    if GRADIENT {
                        let f = -2.0 * w.world * pen * g;
                        force[link] += f;
                        moment[link] += sc[s].cross(f);
                        touched = true;
                    }
                }
            }
        }
    }
    // Likewise for link pairs whose bounding spheres are farther apart than the self margin.
    let gate = w.self_margin.max(0.0);
    for lp in &robot.self_link_pairs {
        let (ba, bb) = (robot.link_bounds[lp.a as usize], robot.link_bounds[lp.b as usize]);
        let at = |link: u32, b: [f32; 4]| rot[link as usize] * Vec3::new(b[0], b[1], b[2]) + pos[link as usize];
        let gap = (at(lp.a, ba) - at(lp.b, bb)).length() - ba[3] - bb[3];
        if gap > gate {
            smin = smin.min(gap);
            continue;
        }
        for &[a, b] in &robot.self_pairs[lp.first as usize..(lp.first + lp.count) as usize] {
            let (a, b) = (a as usize, b as usize);
            let (sa, sb) = (&robot.spheres[a], &robot.spheres[b]);
            let diff = sc[a] - sc[b];
            let dist = diff.length();
            let d = dist - (sa.radius + sa.self_buffer) - (sb.radius + sb.self_buffer);
            smin = smin.min(d);
            let pen = w.self_margin - d;
            if pen > 0.0 && GRADIENT {
                let u = if dist > 1e-9 { diff / dist } else { Vec3::X };
                let g = 2.0 * w.self_collision * pen * u;
                let (la, lb) = (lp.a as usize, lp.b as usize);
                force[la] -= g;
                moment[la] -= sc[a].cross(g);
                force[lb] += g;
                moment[lb] += sc[b].cross(g);
                touched = true;
            }
            if pen > 0.0 {
                cost += w.self_collision * pen * pen;
            }
        }
    }
    if !GRADIENT || !touched {
        return CollisionOut { cost, world_clearance: wmin, self_clearance: smin };
    }
    // Each link's wrench reaches the joints at and above it, leaves first.
    for i in (0..robot.links.len()).rev() {
        let link = &robot.links[i];
        match link.joint {
            JointKind::Revolute { dof, curve, .. } => {
                grad[dof] += fk.rate(i, &curve) * fk.axis[i].dot(moment[i] - pos[i].cross(force[i]));
            }
            JointKind::Prismatic { dof, curve, .. } => grad[dof] += fk.rate(i, &curve) * fk.axis[i].dot(force[i]),
            JointKind::Fixed => {}
        }
        if let Some(p) = link.parent {
            let (f, m) = (force[i], moment[i]);
            force[p] += f;
            moment[p] += m;
        }
    }
    CollisionOut { cost, world_clearance: wmin, self_clearance: smin }
}

/// Rotation vector (axis * angle) of `r`. Must stay in sync with `rot_log` in kernels.wgsl.
fn rot_log(r: Mat3) -> (Vec3, f32) {
    let m = |row: usize, col: usize| r.col(col)[row];
    let cos_t = ((m(0, 0) + m(1, 1) + m(2, 2) - 1.0) * 0.5).clamp(-1.0, 1.0);
    let v = 0.5 * Vec3::new(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));
    let sv = v.length();
    let theta = sv.atan2(cos_t);
    if sv > 1e-6 {
        return (v * (theta / sv), theta);
    }
    if cos_t > 0.0 {
        return (v, theta);
    }
    // Near pi: the axis is the dominant column of (R + I).
    let k = if m(1, 1) > m(0, 0) {
        if m(2, 2) > m(1, 1) { 2 } else { 1 }
    } else if m(2, 2) > m(0, 0) {
        2
    } else {
        0
    };
    ((r.col(k) + Vec3::AXES[k]).normalize() * theta, theta)
}

/// In-place Cholesky factorization of a 6x6 SPD matrix (row-major, lower triangle used).
fn chol6(a: &mut [f32; 36]) {
    for i in 0..6 {
        for j in 0..=i {
            let mut s = a[i * 6 + j];
            for k in 0..j {
                s -= a[i * 6 + k] * a[j * 6 + k];
            }
            a[i * 6 + j] = if i == j { s.max(1e-12).sqrt() } else { s / a[j * 6 + j] };
        }
    }
}

fn chol6_solve(l: &[f32; 36], b: [f32; 6]) -> [f32; 6] {
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * 6 + k] * y[k];
        }
        y[i] = s / l[i * 6 + i];
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in i + 1..6 {
            s -= l[k * 6 + i] * x[k];
        }
        x[i] = s / l[i * 6 + i];
    }
    x
}

/// Pose error of the end effector: (6-vector [position; rot_weight * rotation], |position|, angle).
fn pose_error(robot: &Robot, fk: &Fk, target: &(Vec3, Mat3), rot_weight: f32) -> ([f32; 6], f32, f32) {
    let ee = robot.ee_link;
    let ep = target.0 - fk.pos[ee];
    let (eo, angle) = rot_log(target.1 * fk.rot[ee].transpose());
    let eo = eo * rot_weight;
    ([ep.x, ep.y, ep.z, eo.x, eo.y, eo.z], ep.length(), angle)
}

/// One damped-least-squares IK step with a null-space collision push.
fn ik_step(robot: &Robot, world: &[Prepared], target: &(Vec3, Mat3), q: &mut [f32], o: &IkOptions, s: &mut Scratch) {
    let n = q.len();
    robot.fk_into(q, &mut s.fk);
    let fk = &s.fk;
    let (e, _, _) = pose_error(robot, fk, target, o.rot_weight);
    let ee = robot.ee_link;
    let jac = &mut s.jac[..];
    jac.fill(0.0);
    for (i, dof) in robot.chain(ee) {
        let m = fk.rate(i, &robot.links[i].joint.actuation().expect("chain links have moving joints").1);
        let jp = m * fk.dpoint(i, fk.pos[ee]);
        let jo = if fk.slides[i] { Vec3::ZERO } else { m * fk.axis[i] * o.rot_weight };
        for (r, v) in [jp.x, jp.y, jp.z, jo.x, jo.y, jo.z].into_iter().enumerate() {
            jac[r * n + dof] += v;
        }
    }
    let mut a = [0.0f32; 36];
    for r in 0..6 {
        for c in 0..=r {
            let mut s = 0.0;
            for j in 0..n {
                s += jac[r * n + j] * jac[c * n + j];
            }
            if r == c {
                s += o.damping * o.damping;
            }
            a[r * 6 + c] = s;
            a[c * 6 + r] = s;
        }
    }
    chol6(&mut a);
    let y = chol6_solve(&a, e);
    let dq = &mut s.dq[..];
    dq.fill(0.0);
    for j in 0..n {
        for r in 0..6 {
            dq[j] += jac[r * n + j] * y[r];
        }
    }
    if o.collision_step > 0.0 {
        let g = &mut s.grad[..];
        g.fill(0.0);
        collision::<true>(robot, world, fk, &o.collision, g, &mut s.collision);
        let mut jg = [0.0f32; 6];
        for r in 0..6 {
            for j in 0..n {
                jg[r] += jac[r * n + j] * g[j];
            }
        }
        let z = chol6_solve(&a, jg);
        for j in 0..n {
            let mut proj = g[j];
            for r in 0..6 {
                proj -= jac[r * n + j] * z[r];
            }
            dq[j] -= o.collision_step * proj;
        }
    }
    let largest = dq[..n].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if largest > o.max_step { o.max_step / largest } else { 1.0 };
    for j in 0..n {
        let (lo, hi) = robot.bounds(j);
        q[j] = (q[j] + dq[j] * scale).clamp(lo, hi);
    }
}

/// Collision gradient of the trajectory cost at sample `s` of span `span`, with respect to the
/// configuration there. Must match `traj_samples` in kernels.wgsl.
#[allow(clippy::too_many_arguments)]
fn traj_sample_grad(
    robot: &Robot,
    world: &[Prepared],
    cp: &[f32],
    span: usize,
    s: usize,
    o: &PlanOptions,
    out: &mut [f32],
    scratch: &mut Scratch,
) {
    let n = robot.dof();
    spline::blend(cp, n, span, spline::basis(spline::sample_u(s, o.samples_per_span)), &mut scratch.q);
    robot.fk_into(&scratch.q, &mut scratch.fk);
    out.fill(0.0);
    collision::<true>(robot, world, &scratch.fk, &o.collision, out, &mut scratch.collision);
}

/// Gradient of the trajectory cost w.r.t. free control point `t`: the collision gradients of the
/// samples it shapes, each weighted by its basis value there, plus smoothness on the control
/// points. Must match `traj_grad` in kernels.wgsl.
fn traj_grad(cp: &[f32], sample_grad: &[f32], n: usize, t: usize, o: &PlanOptions, out: &mut [f32]) {
    let t_count = cp.len() / n;
    let (spans, k) = (t_count - 3, o.samples_per_span);
    let q = |i: usize, j: usize| cp[i * n + j];
    out.fill(0.0);
    // Control point t is point i of span t - i.
    for i in 0..4 {
        if t < i || t - i >= spans {
            continue;
        }
        for s in 0..k {
            let w = spline::basis(spline::sample_u(s, k))[i];
            let g = &sample_grad[((t - i) * k + s) * n..((t - i) * k + s + 1) * n];
            for j in 0..n {
                out[j] += w * g[j];
            }
        }
    }
    for j in 0..n {
        let (qm, q0, qp) = (q(t - 1, j), q(t, j), q(t + 1, j));
        let mut g = 2.0 * o.w_vel * (2.0 * q0 - qm - qp);
        g += 2.0 * o.w_acc * -2.0 * (qp - 2.0 * q0 + qm);
        if t >= 2 {
            g += 2.0 * o.w_acc * (q0 - 2.0 * qm + q(t - 2, j));
        }
        if t + 2 < t_count {
            g += 2.0 * o.w_acc * (q(t + 2, j) - 2.0 * qp + q0);
        }
        out[j] += g;
    }
}

/// Control point `t`, joint `j` of the candidate path `x + alpha d`: free points move and are
/// clamped to the joint range, the three pinned at each end stay. Must match `candidate` in
/// kernels.wgsl.
fn candidate(robot: &Robot, x: &[f32], d: &[f32], alpha: f32, t: usize, j: usize) -> f32 {
    let n = robot.dof();
    let i = t * n + j;
    if t < 3 || t >= x.len() / n - 3 {
        return x[i];
    }
    let (lo, hi) = robot.bounds(j);
    (x[i] + alpha * d[i]).clamp(lo, hi)
}

/// Collision cost at sample `s` of span `span` of the candidate path `x + alpha d`. Must match
/// `traj_costs` in kernels.wgsl.
#[allow(clippy::too_many_arguments)]
fn traj_cost(
    robot: &Robot,
    world: &[Prepared],
    x: &[f32],
    d: &[f32],
    alpha: f32,
    span: usize,
    s: usize,
    o: &PlanOptions,
    scratch: &mut Scratch,
) -> f32 {
    let n = robot.dof();
    let w = spline::basis(spline::sample_u(s, o.samples_per_span));
    for j in 0..n {
        let c = |i: usize| candidate(robot, x, d, alpha, span + i, j);
        scratch.q[j] = w[0] * c(0) + w[1] * c(1) + w[2] * c(2) + w[3] * c(3);
    }
    robot.fk_into(&scratch.q, &mut scratch.fk);
    collision::<false>(robot, world, &scratch.fk, &o.collision, &mut [], &mut scratch.collision).cost
}

/// One path's L-BFGS state, as kernels.wgsl keeps it in `aux`.
struct Lbfgs {
    /// Collision cost per line-search step and sample.
    costs: Vec<f32>,
    grad: Vec<f32>,
    prev_grad: Vec<f32>,
    dir: Vec<f32>,
    /// The last accepted step, waiting for the gradient change it caused.
    pending_step: Vec<f32>,
    /// Ring buffers of `history` steps and gradient changes; `newest` is the latest slot.
    steps: Vec<f32>,
    changes: Vec<f32>,
    cost: f32,
    count: usize,
    newest: usize,
    started: bool,
    pending: bool,
}

impl Lbfgs {
    fn new(points: usize, n: usize, samples: usize, o: &PlanOptions) -> Self {
        let tn = points * n;
        Self {
            costs: vec![0.0; LINE_SEARCH.len() * samples],
            grad: vec![0.0; tn],
            prev_grad: vec![0.0; tn],
            dir: vec![0.0; tn],
            pending_step: vec![0.0; tn],
            steps: vec![0.0; o.history * tn],
            changes: vec![0.0; o.history * tn],
            cost: 0.0,
            count: 0,
            newest: 0,
            started: false,
            pending: false,
        }
    }
}

/// Line search: the trajectory cost of each step along the direction; the cheapest moves the path
/// if it lowers the cost, otherwise the history resets. Must match `traj_search` in kernels.wgsl.
fn traj_search(robot: &Robot, x: &mut [f32], st: &mut Lbfgs, o: &PlanOptions) {
    let n = robot.dof();
    let t_count = x.len() / n;
    let samples = st.costs.len() / LINE_SEARCH.len();
    let mut best = None;
    for (c, &alpha) in LINE_SEARCH.iter().enumerate() {
        let mut total = 0.0;
        for s in 0..samples {
            total += st.costs[c * samples + s];
        }
        for j in 0..n {
            // Each candidate point once, sliding along the path: (before, previous, current).
            let (mut before, mut previous) = (0.0, candidate(robot, x, &st.dir, alpha, 0, j));
            for t in 1..t_count {
                let current = candidate(robot, x, &st.dir, alpha, t, j);
                let v = current - previous;
                total += o.w_vel * v * v;
                if t >= 2 {
                    let a = current - 2.0 * previous + before;
                    total += o.w_acc * a * a;
                }
                (before, previous) = (previous, current);
            }
        }
        if (best.is_none() && !st.started) || total < st.cost {
            st.cost = total;
            best = Some(alpha);
        }
    }
    let Some(alpha) = best else {
        st.pending = false;
        st.count = 0;
        return;
    };
    for t in 3..t_count - 3 {
        for j in 0..n {
            let i = t * n + j;
            let moved = candidate(robot, x, &st.dir, alpha, t, j);
            st.pending_step[i] = moved - x[i];
            x[i] = moved;
        }
    }
    st.started = true;
    st.pending = true;
}

/// Records the last step and the gradient change it caused, then the L-BFGS two-loop recursion
/// for the next direction; steepest descent scaled to `initial_step` without history or when
/// the recursion fails to descend. Must match `lbfgs_direction` in kernels.wgsl.
fn lbfgs_direction(st: &mut Lbfgs, n: usize, o: &PlanOptions) {
    let tn = st.grad.len();
    let free = 3 * n..tn - 3 * n;
    let m = o.history;
    if st.pending {
        let mut sy = 0.0;
        for i in free.clone() {
            sy += st.pending_step[i] * (st.grad[i] - st.prev_grad[i]);
        }
        if sy > 1e-10 {
            let slot = if st.count == 0 { 0 } else { (st.newest + 1) % m };
            for i in free.clone() {
                st.steps[slot * tn + i] = st.pending_step[i];
                st.changes[slot * tn + i] = st.grad[i] - st.prev_grad[i];
            }
            st.newest = slot;
            st.count = (st.count + 1).min(m);
        }
        st.pending = false;
    }
    let mut largest = 0.0f32;
    for i in free.clone() {
        st.prev_grad[i] = st.grad[i];
        st.dir[i] = -st.grad[i];
        largest = largest.max(st.grad[i].abs());
    }
    let dot = |a: &[f32], b: &[f32]| free.clone().map(|i| a[i] * b[i]).sum::<f32>();
    let mut alpha = [0.0f32; MAX_HISTORY];
    for a in 0..st.count {
        let slot = (st.newest + m - a) % m;
        let (s, y) = (&st.steps[slot * tn..(slot + 1) * tn], &st.changes[slot * tn..(slot + 1) * tn]);
        alpha[a] = dot(s, &st.dir) / dot(y, s);
        for i in free.clone() {
            st.dir[i] -= alpha[a] * y[i];
        }
    }
    let gamma = if st.count > 0 {
        let slot = st.newest;
        let (s, y) = (&st.steps[slot * tn..(slot + 1) * tn], &st.changes[slot * tn..(slot + 1) * tn]);
        dot(s, y) / dot(y, y)
    } else if largest > 0.0 {
        o.initial_step / largest
    } else {
        0.0
    };
    for i in free.clone() {
        st.dir[i] *= gamma;
    }
    for a in (0..st.count).rev() {
        let slot = (st.newest + m - a) % m;
        let (s, y) = (&st.steps[slot * tn..(slot + 1) * tn], &st.changes[slot * tn..(slot + 1) * tn]);
        let beta = dot(y, &st.dir) / dot(y, s);
        for i in free.clone() {
            st.dir[i] += (alpha[a] - beta) * s[i];
        }
    }
    if largest > 0.0 && dot(&st.dir, &st.grad) >= 0.0 {
        st.count = 0;
        for i in free.clone() {
            st.dir[i] = -st.grad[i] * (o.initial_step / largest);
        }
    }
}

/// `f(scratch, index, chunk)` over `size`-long chunks of `data`: on the pool's threads, each with
/// its own scratch, when `parallel`; otherwise in order with `scratch`.
fn each_chunk<T: Send>(
    parallel: bool,
    robot: &Robot,
    scratch: &mut Scratch,
    data: &mut [T],
    size: usize,
    f: impl Fn(&mut Scratch, usize, &mut [T]) + Sync,
) {
    if parallel {
        data.par_chunks_mut(size).enumerate().for_each_init(|| Scratch::new(robot), |s, (i, chunk)| f(s, i, chunk));
    } else {
        data.chunks_mut(size).enumerate().for_each(|(i, chunk)| f(scratch, i, chunk));
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("cpu ({} threads)", self.pool.current_num_threads())
    }

    fn robot(&self) -> &Robot {
        &self.robot
    }

    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(prepare(worlds)))
    }

    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>> {
        Ok(Box::new(Self { robot: robot.clone(), pool: self.pool.clone() }))
    }

    fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation> {
        let n = self.robot.dof();
        let prepared: &CpuWorlds = worlds.prepared();
        let items = item_world.len();
        let mut out = Evaluation {
            world_clearance: vec![0.0; items],
            self_clearance: vec![0.0; items],
            cost: vec![0.0; items],
            grad: vec![0.0; items * n],
        };
        let rows =
            out.world_clearance.par_iter_mut().zip(out.self_clearance.par_iter_mut()).zip(out.cost.par_iter_mut());
        self.pool.install(|| {
            rows.zip(out.grad.par_chunks_mut(n)).zip(q.par_chunks(n).zip(item_world.par_iter())).for_each_init(
                || Scratch::new(&self.robot),
                |scratch, ((((wc, sc), cost), g), (qi, &s))| {
                    self.robot.fk_into(qi, &mut scratch.fk);
                    let world = &prepared[s as usize];
                    let c = collision::<true>(&self.robot, world, &scratch.fk, w, g, &mut scratch.collision);
                    (*wc, *sc, *cost) = (c.world_clearance, c.self_clearance, c.cost);
                },
            )
        });
        Ok(out)
    }

    fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>> {
        let n = self.robot.dof();
        let prepared: &CpuWorlds = worlds.prepared();
        let mut out = vec![[0.0; 2]; item_world.len()];
        self.pool.install(|| {
            out.par_iter_mut().zip(q.par_chunks(n).zip(item_world.par_iter())).for_each_init(
                || Scratch::new(&self.robot),
                |scratch, (c, (qi, &s))| {
                    self.robot.fk_into(qi, &mut scratch.fk);
                    let world = &prepared[s as usize];
                    let none = &CollisionWeights::NONE;
                    let r = collision::<false>(&self.robot, world, &scratch.fk, none, &mut [], &mut scratch.collision);
                    *c = [r.world_clearance, r.self_clearance];
                },
            )
        });
        Ok(out)
    }

    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        let n = self.robot.dof();
        let prepared: &CpuWorlds = worlds.prepared();
        Ok(self.pool.install(|| {
            q.par_chunks_mut(n)
                .zip(item_world.par_iter().zip(targets.par_iter()))
                .map_init(
                    || Scratch::new(&self.robot),
                    |scratch, (qi, (&s, target))| {
                        let world = &prepared[s as usize];
                        let target = (target.position, Mat3::from_quat(target.rotation));
                        for _ in 0..o.iterations {
                            ik_step(&self.robot, world, &target, qi, o, scratch);
                        }
                        self.robot.fk_into(qi, &mut scratch.fk);
                        let (_, pos_err, rot_err) = pose_error(&self.robot, &scratch.fk, &target, o.rot_weight);
                        [pos_err, rot_err]
                    },
                )
                .collect()
        }))
    }

    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let n = self.robot.dof();
        let t_count = paths.points;
        let samples = (t_count - 3) * o.samples_per_span;
        let prepared: &CpuWorlds = worlds.prepared();
        if o.iterations == 0 {
            return Ok(());
        }
        // Fewer paths than threads (batch-1 planning): spread each path's samples over threads too.
        let per_sample = item_world.len() < self.pool.current_num_threads();
        let optimize = |(tr, &s): (&mut [f32], &u32)| {
            let (world, k) = (&prepared[s as usize], o.samples_per_span);
            let mut sample_grad = vec![0.0f32; samples * n];
            let mut st = Lbfgs::new(t_count, n, samples, o);
            let mut scratch = Scratch::new(&self.robot);
            // The first round only prices the seed: its direction is still zero.
            for round in 0..=o.iterations {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let tr_ref = &*tr;
                let dir = &st.dir;
                each_chunk(per_sample, &self.robot, &mut scratch, &mut st.costs, 1, |s, i, cost| {
                    let (c, i) = (i / samples, i % samples);
                    cost[0] = traj_cost(&self.robot, world, tr_ref, dir, LINE_SEARCH[c], i / k, i % k, o, s);
                });
                let steepest = st.count == 0;
                traj_search(&self.robot, tr, &mut st, o);
                // A failed steepest-descent search leaves nothing to change in later rounds.
                if round == o.iterations || (steepest && st.started && !st.pending) {
                    break;
                }
                let tr_ref = &*tr;
                each_chunk(per_sample, &self.robot, &mut scratch, &mut sample_grad, n, |s, i, g| {
                    traj_sample_grad(&self.robot, world, tr_ref, i / k, i % k, o, g, s);
                });
                for t in 3..t_count - 3 {
                    traj_grad(tr, &sample_grad, n, t, o, &mut st.grad[t * n..(t + 1) * n]);
                }
                lbfgs_direction(&mut st, n, o);
            }
        };
        self.pool.install(|| paths.positions.par_chunks_mut(t_count * n).zip(item_world.par_iter()).for_each(optimize));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn lbfgs_direction_applies_the_bfgs_inverse_hessian() {
        // Three remembered steps in a ring of four, newest in slot 0 after wrapping around; the
        // two-loop recursion must equal -H g with H built by explicit BFGS updates.
        let (n, points) = (2, 10);
        let o = PlanOptions { history: 4, ..Default::default() };
        let mut st = Lbfgs::new(points, n, 1, &o);
        let free: Vec<usize> = (3 * n..(points - 3) * n).collect();
        let mut rng = Rng::new(1);
        let tn = points * n;
        let mut pairs = vec![];
        for (age, slot) in [(0, 2), (1, 3), (2, 0)] {
            let s: Vec<f32> = (0..tn).map(|i| if free.contains(&i) { rng.range(-1.0, 1.0) } else { 0.0 }).collect();
            // y = A s for a fixed positive diagonal A, so s.y > 0.
            let y: Vec<f32> = s.iter().enumerate().map(|(i, v)| v * (1.0 + i as f32 * 0.1)).collect();
            st.steps[slot * tn..(slot + 1) * tn].copy_from_slice(&s);
            st.changes[slot * tn..(slot + 1) * tn].copy_from_slice(&y);
            pairs.push((age, s, y));
        }
        (st.count, st.newest) = (3, 0);
        for &i in &free {
            st.grad[i] = rng.range(-1.0, 1.0);
        }
        lbfgs_direction(&mut st, n, &o);
        // H = gamma I, then each pair oldest first: H <- (I - r s y') H (I - r y s') + r s s'.
        let k = free.len();
        let (_, s_new, y_new) = &pairs[2];
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let pick = |v: &[f32]| free.iter().map(|&i| v[i] as f64).collect::<Vec<f64>>();
        let gamma = dot(&pick(s_new), &pick(y_new)) / dot(&pick(y_new), &pick(y_new));
        let mut h: Vec<f64> = (0..k * k).map(|i| if i % (k + 1) == 0 { gamma } else { 0.0 }).collect();
        for (_, s, y) in &pairs {
            let (s, y) = (pick(s), pick(y));
            let r = 1.0 / dot(&s, &y);
            let left: Vec<f64> = (0..k * k).map(|i| f64::from(i / k == i % k) - r * s[i / k] * y[i % k]).collect();
            let mul = |a: &[f64], b: &[f64]| -> Vec<f64> {
                (0..k * k).map(|i| (0..k).map(|m| a[i / k * k + m] * b[m * k + i % k]).sum()).collect()
            };
            let right: Vec<f64> = (0..k * k).map(|i| left[i % k * k + i / k]).collect();
            h = mul(&mul(&left, &h), &right);
            for i in 0..k * k {
                h[i] += r * s[i / k] * s[i % k];
            }
        }
        let g = pick(&st.grad);
        for (row, &i) in free.iter().enumerate() {
            let expected = -(0..k).map(|c| h[row * k + c] * g[c]).sum::<f64>();
            assert!(
                (st.dir[i] as f64 - expected).abs() < 1e-4 * expected.abs().max(1.0),
                "{} vs {expected}",
                st.dir[i]
            );
        }
    }

    #[test]
    fn a_failed_line_search_resets_the_history() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/franka");
        let spheres = crate::robot::CollisionModel::load(format!("{dir}/panda_collision.json")).unwrap();
        let options = crate::robot::RobotOptions { collision_model: Some(spheres), ..Default::default() };
        let robot = Robot::load(format!("{dir}/franka_panda.urdf"), &options).unwrap();
        let (n, points) = (robot.dof(), 10);
        let o = PlanOptions { control_points: points, ..Default::default() };
        let mut st = Lbfgs::new(points, n, 1, &o);
        let mut x = vec![0.0; points * n];
        (st.started, st.cost, st.count) = (true, 1.0, 3);
        st.costs.fill(2.0);
        traj_search(&robot, &mut x, &mut st, &o);
        assert_eq!((st.count, st.pending), (0, false), "a step that raises the cost must not be taken");
        assert!(x.iter().all(|&v| v == 0.0));
        st.costs.fill(0.0);
        st.count = 3;
        traj_search(&robot, &mut x, &mut st, &o);
        assert!(st.pending && st.count == 3 && st.cost < 1.0, "a cheaper step must be taken");
    }
}
