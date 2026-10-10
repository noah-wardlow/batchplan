//! Batched trajectory optimization: many seeds per start/goal pair, each a uniform cubic B-spline
//! whose control points Adam optimizes on smoothness plus collision cost sampled along the curve,
//! then validated by sampling the curve densely. The first three and last three control points
//! are pinned to the start and goal, so every path starts and ends at rest.

use anyhow::{Result, ensure};

use crate::device::{CollisionWeights, Device};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::spline;
use crate::types::{JointPaths, Solved};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub seeds: usize,
    /// B-spline control points per path, including the three pinned at each end (at least 7).
    pub control_points: usize,
    /// Collision samples per span during optimization.
    pub samples_per_span: usize,
    pub iterations: u32,
    pub learning_rate: f32,
    pub lr_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    /// Adam's denominator offset; gradients far below it take proportionally small steps.
    pub adam_epsilon: f32,
    /// Weight of squared second differences of control points.
    pub w_acc: f32,
    /// Weight of squared differences of control points (path length).
    pub w_vel: f32,
    pub collision: CollisionWeights,
    /// Samples per span used to validate the result.
    pub validate_substeps: usize,
    pub rng_seed: u64,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            seeds: 8,
            control_points: 24,
            samples_per_span: 3,
            iterations: 200,
            learning_rate: 0.03,
            lr_decay: 0.99,
            beta1: 0.9,
            beta2: 0.999,
            adam_epsilon: 1e-8,
            w_acc: 50.0,
            w_vel: 2.0,
            collision: CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.02, self_margin: 0.01 },
            validate_substeps: 8,
            rng_seed: 2,
        }
    }
}

impl PlanOptions {
    /// Adam learning rate and bias corrections for iteration `k` (shared by all backends).
    pub(crate) fn schedule(&self, k: u32) -> [f32; 3] {
        let lr = self.learning_rate * self.lr_decay.powi(k as i32);
        [lr, 1.0 - self.beta1.powi(k as i32 + 1), 1.0 - self.beta2.powi(k as i32 + 1)]
    }
}

#[derive(Clone, Debug)]
pub struct PlanProblem {
    pub world: u32,
    pub start: Vec<f32>,
    pub goal: Vec<f32>,
}

/// Per-seed results for `problems`; paths are problem-major (`item = problem * seeds + seed`).
#[derive(Clone, Debug)]
pub struct PlanResult {
    pub problems: Vec<PlanProblem>,
    pub seeds: usize,
    /// B-spline control points of every seed's path.
    pub paths: JointPaths,
    pub valid: Vec<bool>,
    /// Minimum of world and self clearance along the densely sampled path.
    pub min_clearance: Vec<f32>,
    /// Joint-space path length.
    pub length: Vec<f32>,
}

impl PlanResult {
    /// The shortest valid seed.
    pub fn best(&self, problem: usize) -> Option<&[f32]> {
        (problem * self.seeds..(problem + 1) * self.seeds)
            .filter(|&i| self.valid[i])
            .min_by(|&a, &b| self.length[a].total_cmp(&self.length[b]))
            .map(|i| self.paths.path(i))
    }

    /// Every problem with a valid seed, with its best path's control points (`[points, dof]`);
    /// [`crate::timing::Trajectory::new`] times one for execution.
    pub fn solved(&self) -> impl Iterator<Item = Solved<'_, PlanProblem>> {
        self.problems
            .iter()
            .enumerate()
            .filter_map(|(index, problem)| Some(Solved { index, problem, solution: self.best(index)? }))
    }
}

pub fn plan(device: &Device, worlds: &[World], problems: &[PlanProblem], o: &PlanOptions) -> Result<PlanResult> {
    let robot = device.robot();
    let n = robot.dof();
    let points = o.control_points;
    ensure!(points >= 7 && o.seeds > 0, "need at least 7 control points and one seed");
    ensure!(o.samples_per_span > 0, "need at least one collision sample per span");
    let items = problems.len() * o.seeds;
    let mut rng = Rng::new(o.rng_seed);
    let mut paths = JointPaths::zeros(items, points, n);
    let mut item_world = Vec::with_capacity(items);
    for (pi, p) in problems.iter().enumerate() {
        ensure!(p.start.len() == n && p.goal.len() == n, "problem {pi}: start/goal must have {n} values");
        for s in 0..o.seeds {
            seed_path(robot, &p.start, &p.goal, s, &mut rng, paths.path_mut(pi * o.seeds + s));
            item_world.push(p.world);
        }
    }
    device.trajopt(worlds, &item_world, &mut paths, o)?;

    // Validate along the curve itself.
    let k = o.validate_substeps.max(1);
    let samples = (points - 3) * k + 1;
    let mut dense = Vec::with_capacity(items * samples * n);
    for item in 0..items {
        dense.extend(spline::dense(paths.path(item), n, k));
    }
    let dense_world: Vec<u32> = item_world.iter().flat_map(|&w| std::iter::repeat_n(w, samples)).collect();
    let eval = device.evaluate(worlds, &dense_world, &dense, &CollisionWeights::NONE)?;
    let min_clearance: Vec<f32> = (0..items)
        .map(|item| {
            (item * samples..(item + 1) * samples)
                .map(|i| eval.world_clearance[i].min(eval.self_clearance[i]))
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    let length = (0..items)
        .map(|item| {
            let q = &dense[item * samples * n..(item + 1) * samples * n];
            q.chunks(n)
                .zip(q.chunks(n).skip(1))
                .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (y - x).powi(2)).sum::<f32>().sqrt())
                .sum()
        })
        .collect();
    Ok(PlanResult {
        problems: problems.to_vec(),
        seeds: o.seeds,
        paths,
        valid: min_clearance.iter().map(|&c| c >= 0.0).collect(),
        min_clearance,
        length,
    })
}

/// Control points of seed 0 lie on the straight line; other seeds bend through a random via point.
/// The first three and last three are the start and goal.
fn seed_path(robot: &Robot, start: &[f32], goal: &[f32], seed: usize, rng: &mut Rng, out: &mut [f32]) {
    let n = start.len();
    let t_count = out.len() / n;
    let (via_u, via): (f32, Vec<f32>) = if seed == 0 {
        (0.5, (0..n).map(|j| 0.5 * (start[j] + goal[j])).collect())
    } else {
        let u = rng.range(0.3, 0.7);
        let alpha = rng.range(0.2, 0.6);
        let via = (0..n)
            .map(|j| {
                let mid = start[j] + u * (goal[j] - start[j]);
                mid + alpha * (rng.range(robot.lower[j], robot.upper[j]) - mid)
            })
            .collect();
        (u, via)
    };
    // Free control points 3..t_count - 3 spread evenly along the seed between the pinned ends.
    for t in 3..t_count - 3 {
        let u = (t - 2) as f32 / (t_count - 5) as f32;
        for j in 0..n {
            out[t * n + j] = if u <= via_u {
                start[j] + (via[j] - start[j]) * (u / via_u)
            } else {
                via[j] + (goal[j] - via[j]) * ((u - via_u) / (1.0 - via_u))
            };
        }
    }
    // Copied, not interpolated, so they stay bit-exact.
    for t in 0..3 {
        out[t * n..(t + 1) * n].copy_from_slice(start);
        out[(t_count - 1 - t) * n..(t_count - t) * n].copy_from_slice(goal);
    }
}
