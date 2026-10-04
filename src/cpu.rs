//! CPU reference backend (rayon over items). Every function here has a WGSL twin in
//! `kernels.wgsl`; the GPU parity tests compare the two.

// Index loops here mirror kernels.wgsl line for line, which keeps the two easy to compare.
#![allow(clippy::needless_range_loop)]

use anyhow::Result;
use glam::{Mat3, Vec3};
use rayon::prelude::*;

use crate::device::{Backend, CollisionWeights, Evaluation};
use crate::ik::IkOptions;
use crate::robot::{Fk, MAX_DOF, MAX_SPHERES, Robot};
use crate::trajopt::PlanOptions;
use crate::types::{JointPaths, Pose};
use crate::world::{FAR, Obstacle, World, box_distance, sphere_distance};

pub(crate) struct CpuBackend {
    robot: Robot,
}

impl CpuBackend {
    pub(crate) fn new(robot: &Robot) -> Self {
        Self { robot: robot.clone() }
    }
}

/// Obstacle with its rotation matrix precomputed.
#[derive(Clone, Copy)]
enum Prepared {
    Cuboid { rot: Mat3, center: Vec3, half: Vec3 },
    Sphere { center: Vec3, radius: f32 },
}

fn prepare(worlds: &[World]) -> Vec<Vec<Prepared>> {
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

/// Collision cost of the configuration behind `fk`; adds d(cost)/dq into `grad`.
fn collision(robot: &Robot, world: &[Prepared], fk: &Fk, w: &CollisionWeights, grad: &mut [f32]) -> CollisionOut {
    let ns = robot.spheres.len();
    let mut sc = [Vec3::ZERO; MAX_SPHERES];
    let mut gc = [Vec3::ZERO; MAX_SPHERES];
    for (s, sp) in robot.spheres.iter().enumerate() {
        sc[s] = fk.rot[sp.link] * sp.center + fk.pos[sp.link];
    }
    let (mut cost, mut wmin, mut smin) = (0.0f32, FAR, FAR);
    for s in 0..ns {
        let r = robot.spheres[s].radius;
        for o in world {
            let (dist, g) = match *o {
                Prepared::Cuboid { rot, center, half } => box_distance(rot, center, half, sc[s]),
                Prepared::Sphere { center, radius } => sphere_distance(center, radius, sc[s]),
            };
            let d = dist - r;
            wmin = wmin.min(d);
            let pen = w.margin - d;
            if pen > 0.0 {
                cost += w.world * pen * pen;
                gc[s] -= 2.0 * w.world * pen * g;
            }
        }
    }
    for &[a, b] in &robot.self_pairs {
        let (a, b) = (a as usize, b as usize);
        let (sa, sb) = (&robot.spheres[a], &robot.spheres[b]);
        let diff = sc[a] - sc[b];
        let dist = diff.length();
        let d = dist - (sa.radius + sa.self_buffer) - (sb.radius + sb.self_buffer);
        smin = smin.min(d);
        let pen = w.self_margin - d;
        if pen > 0.0 {
            cost += w.self_collision * pen * pen;
            let u = if dist > 1e-9 { diff / dist } else { Vec3::X };
            let g = 2.0 * w.self_collision * pen * u;
            gc[a] -= g;
            gc[b] += g;
        }
    }
    for s in 0..ns {
        if gc[s] == Vec3::ZERO {
            continue;
        }
        let mask = robot.links[robot.spheres[s].link].dof_mask;
        for (j, gj) in grad.iter_mut().enumerate() {
            if mask >> j & 1 == 1 {
                *gj += gc[s].dot(fk.dpoint(j, sc[s]));
            }
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
fn ik_step(robot: &Robot, world: &[Prepared], target: &(Vec3, Mat3), q: &mut [f32], o: &IkOptions) {
    let n = q.len();
    let fk = robot.fk(q);
    let (e, _, _) = pose_error(robot, &fk, target, o.rot_weight);
    let ee = robot.ee_link;
    let mask = robot.links[ee].dof_mask;
    let mut jac = [[0.0f32; MAX_DOF]; 6];
    for j in 0..n {
        if mask >> j & 1 == 1 {
            let jp = fk.dpoint(j, fk.pos[ee]);
            let jo = if fk.prismatic >> j & 1 == 1 { Vec3::ZERO } else { fk.axis[j] * o.rot_weight };
            for (r, v) in [jp.x, jp.y, jp.z, jo.x, jo.y, jo.z].into_iter().enumerate() {
                jac[r][j] = v;
            }
        }
    }
    let mut a = [0.0f32; 36];
    for r in 0..6 {
        for c in 0..=r {
            let mut s = 0.0;
            for j in 0..n {
                s += jac[r][j] * jac[c][j];
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
    let mut dq = [0.0f32; MAX_DOF];
    for j in 0..n {
        for r in 0..6 {
            dq[j] += jac[r][j] * y[r];
        }
    }
    if o.collision_step > 0.0 {
        let mut g = [0.0f32; MAX_DOF];
        collision(robot, world, &fk, &o.collision, &mut g[..n]);
        let mut jg = [0.0f32; 6];
        for r in 0..6 {
            for j in 0..n {
                jg[r] += jac[r][j] * g[j];
            }
        }
        let z = chol6_solve(&a, jg);
        for j in 0..n {
            let mut proj = g[j];
            for r in 0..6 {
                proj -= jac[r][j] * z[r];
            }
            dq[j] -= o.collision_step * proj;
        }
    }
    let largest = dq[..n].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if largest > o.max_step { o.max_step / largest } else { 1.0 };
    for j in 0..n {
        q[j] = (q[j] + dq[j] * scale).clamp(robot.lower[j], robot.upper[j]);
    }
}

/// Gradient of the trajectory cost w.r.t. interior waypoint `t`. Must match `traj_grad` in kernels.wgsl.
fn traj_grad(robot: &Robot, world: &[Prepared], tr: &[f32], t: usize, o: &PlanOptions, out: &mut [f32]) {
    let n = robot.dof();
    let t_count = tr.len() / n;
    let q = |k: usize, j: usize| tr[k * n + j];
    out.fill(0.0);
    let fk = robot.fk(&tr[t * n..(t + 1) * n]);
    collision(robot, world, &fk, &o.collision, out);
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

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn robot(&self) -> &Robot {
        &self.robot
    }

    fn evaluate(&self, worlds: &[World], item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation> {
        let n = self.robot.dof();
        let prepared = prepare(worlds);
        let rows: Vec<(f32, f32, f32, Vec<f32>)> = q
            .par_chunks(n)
            .zip(item_world.par_iter())
            .map(|(qi, &s)| {
                let mut g = vec![0.0; n];
                let c = collision(&self.robot, &prepared[s as usize], &self.robot.fk(qi), w, &mut g);
                (c.world_clearance, c.self_clearance, c.cost, g)
            })
            .collect();
        let mut out = Evaluation::default();
        for (wc, sc, cost, g) in rows {
            out.world_clearance.push(wc);
            out.self_clearance.push(sc);
            out.cost.push(cost);
            out.grad.extend(g);
        }
        Ok(out)
    }

    fn ik(
        &self,
        worlds: &[World],
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        let n = self.robot.dof();
        let prepared = prepare(worlds);
        Ok(q.par_chunks_mut(n)
            .zip(item_world.par_iter().zip(targets.par_iter()))
            .map(|(qi, (&s, target))| {
                let world = &prepared[s as usize];
                let target = (target.position, Mat3::from_quat(target.rotation));
                for _ in 0..o.iterations {
                    ik_step(&self.robot, world, &target, qi, o);
                }
                let (_, pos_err, rot_err) = pose_error(&self.robot, &self.robot.fk(qi), &target, o.rot_weight);
                [pos_err, rot_err]
            })
            .collect())
    }

    fn trajopt(&self, worlds: &[World], item_world: &[u32], paths: &mut JointPaths, o: &PlanOptions) -> Result<()> {
        let n = self.robot.dof();
        let t_count = paths.waypoints;
        let prepared = prepare(worlds);
        paths.positions.par_chunks_mut(t_count * n).zip(item_world.par_iter()).for_each(|(tr, &s)| {
            let world = &prepared[s as usize];
            let mut grad = vec![0.0f32; t_count * n];
            let mut m = vec![0.0f32; t_count * n];
            let mut v = vec![0.0f32; t_count * n];
            for k in 0..o.iterations {
                for t in 1..t_count - 1 {
                    traj_grad(&self.robot, world, tr, t, o, &mut grad[t * n..(t + 1) * n]);
                }
                let [lr, bc1, bc2] = o.schedule(k);
                for t in 1..t_count - 1 {
                    for j in 0..n {
                        let i = t * n + j;
                        let g = grad[i];
                        m[i] = o.beta1 * m[i] + (1.0 - o.beta1) * g;
                        v[i] = o.beta2 * v[i] + (1.0 - o.beta2) * g * g;
                        let step = lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + o.adam_epsilon);
                        tr[i] = (tr[i] - step).clamp(self.robot.lower[j], self.robot.upper[j]);
                    }
                }
            }
        });
        Ok(())
    }
}
