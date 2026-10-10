//! Batched trajectory optimization: many seeds per start/goal pair, each a uniform cubic B-spline
//! whose control points L-BFGS optimizes on smoothness plus collision cost sampled along the
//! curve, then validated by sampling the curve densely. Each L-BFGS step prices four fractions of
//! its direction at once and takes the cheapest. The first three and last three control points
//! are pinned to the start and goal, so every path starts and ends at rest. Problems no seed
//! solves fall back to RRT-Connect: its shortcut path, traced by a B-spline and optimized again.

use std::time::{Duration, Instant};

use crate::error::{Result, ensure_input};

use crate::device::{CollisionWeights, Device, Worlds};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::rrt::{RrtOptions, RrtProblem, connect_until, distance};
use crate::shortcut::{ShortcutOptions, length, locate, shortcut_until};
use crate::spline;
use crate::types::{JointPaths, Solved};

#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub seeds: usize,
    /// B-spline control points per path, including the three pinned at each end (at least 7).
    pub control_points: usize,
    /// Collision samples per span during optimization.
    pub samples_per_span: usize,
    /// L-BFGS steps.
    pub iterations: u32,
    /// Steps L-BFGS remembers (1 to [`MAX_HISTORY`]).
    pub history: usize,
    /// Largest joint change of a step without history: the first, and any after a reset.
    pub initial_step: f32,
    /// Weight of squared second differences of control points.
    pub w_acc: f32,
    /// Weight of squared differences of control points (path length).
    pub w_vel: f32,
    pub collision: CollisionWeights,
    /// Samples per span used to validate the result.
    pub validate_substeps: usize,
    pub rng_seed: u64,
    /// What to do for problems no seed solves; `None` leaves them unsolved.
    pub fallback: Option<Fallback>,
    /// Stop optimizing and searching once this much time has passed, and return the best valid
    /// paths so far. The budget is checked between rounds (on a GPU, between submissions of
    /// several rounds), and validation still runs after it. Results then depend on machine
    /// speed; without a budget they are the same on every run.
    pub time_budget: Option<Duration>,
}

/// Fractions of the L-BFGS direction each line search tries, in order of preference on ties.
pub(crate) const LINE_SEARCH: [f32; 4] = [1.0, 0.5, 0.25, 0.1];
/// The most steps L-BFGS can remember.
pub const MAX_HISTORY: usize = 8;

/// Searching for a path with RRT-Connect, then shortcutting it, before refitting it as a B-spline.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fallback {
    pub rrt: RrtOptions,
    pub shortcut: ShortcutOptions,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            seeds: 8,
            control_points: 24,
            samples_per_span: 3,
            iterations: 40,
            history: 6,
            initial_step: 0.1,
            w_acc: 50.0,
            w_vel: 2.0,
            collision: CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.02, self_margin: 0.01 },
            validate_substeps: 8,
            rng_seed: 2,
            fallback: Some(Fallback::default()),
            time_budget: None,
        }
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

pub fn plan(device: &Device, worlds: &Worlds, problems: &[PlanProblem], o: &PlanOptions) -> Result<PlanResult> {
    let deadline = o.time_budget.map(|budget| Instant::now() + budget);
    let robot = device.robot();
    let n = robot.dof();
    let points = o.control_points;
    ensure_input!(points >= 7 && o.seeds > 0, "need at least 7 control points and one seed");
    ensure_input!(o.samples_per_span > 0, "need at least one collision sample per span");
    ensure_input!((1..=MAX_HISTORY).contains(&o.history), "L-BFGS history must be 1 to {MAX_HISTORY} steps");
    ensure_input!(o.initial_step > 0.0, "the initial step must be positive");
    let items = problems.len() * o.seeds;
    let mut rng = Rng::new(o.rng_seed);
    let mut paths = JointPaths::zeros(items, points, n);
    let mut item_world = Vec::with_capacity(items);
    for (pi, p) in problems.iter().enumerate() {
        ensure_input!(p.start.len() == n && p.goal.len() == n, "problem {pi}: start/goal must have {n} values");
        let within = |q: &[f32]| q.iter().enumerate().all(|(j, &v)| v >= robot.lower[j] && v <= robot.upper[j]);
        ensure_input!(
            within(&p.start) && within(&p.goal),
            "problem {pi}: start and goal must be within the joint limits"
        );
        for s in 0..o.seeds {
            seed_path(robot, &p.start, &p.goal, s, &mut rng, paths.path_mut(pi * o.seeds + s));
            item_world.push(p.world);
        }
    }
    device.trajopt(worlds, &item_world, &mut paths, o, deadline)?;
    let (min_clearance, length) = validate(device, worlds, &paths, &item_world, o)?;
    let mut result = PlanResult {
        problems: problems.to_vec(),
        seeds: o.seeds,
        paths,
        valid: min_clearance.iter().map(|&c| c >= 0.0).collect(),
        min_clearance,
        length,
    };
    if let Some(fallback) = &o.fallback {
        fall_back(device, worlds, &mut result, fallback, o, deadline)?;
    }
    Ok(result)
}

/// The smallest clearance along each path, sampled densely along the curve, and its length.
fn validate(
    device: &Device,
    worlds: &Worlds,
    paths: &JointPaths,
    item_world: &[u32],
    o: &PlanOptions,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (n, items) = (paths.dof, paths.len());
    let k = o.validate_substeps.max(1);
    let samples = (paths.points - 3) * k + 1;
    let mut dense = vec![0.0; items * samples * n];
    for (item, out) in dense.chunks_mut(samples * n).enumerate() {
        spline::dense(paths.path(item), n, k, out);
    }
    let dense_world: Vec<u32> = item_world.iter().flat_map(|&w| std::iter::repeat_n(w, samples)).collect();
    let eval = device.evaluate(worlds, &dense_world, &dense, &CollisionWeights::NONE)?;
    let min_clearance = (0..items)
        .map(|item| {
            (item * samples..(item + 1) * samples)
                .map(|i| eval.world_clearance[i].min(eval.self_clearance[i]))
                // NaN marks the path invalid rather than vanishing in a minimum.
                .fold(f32::INFINITY, |m, c| if m.is_nan() || c.is_nan() { f32::NAN } else { m.min(c) })
        })
        .collect();
    let lengths = (0..items).map(|item| length(&dense[item * samples * n..(item + 1) * samples * n], n)).collect();
    Ok((min_clearance, lengths))
}

/// Gives each problem without a valid seed an RRT-Connect path, shortcut, traced by a B-spline and
/// then optimized. The better valid one of the refit and the
/// optimized spline replaces the problem's first seed.
fn fall_back(
    device: &Device,
    worlds: &Worlds,
    result: &mut PlanResult,
    f: &Fallback,
    o: &PlanOptions,
    deadline: Option<Instant>,
) -> Result<()> {
    let failed: Vec<usize> = (0..result.problems.len()).filter(|&p| result.best(p).is_none()).collect();
    if failed.is_empty() || deadline.is_some_and(|d| Instant::now() >= d) {
        return Ok(());
    }
    let searches: Vec<RrtProblem> = failed
        .iter()
        .map(|&p| {
            let problem = &result.problems[p];
            RrtProblem { world: problem.world, start: problem.start.clone(), goals: vec![problem.goal.clone()] }
        })
        .collect();
    let found = connect_until(device, worlds, &searches, &f.rrt, deadline)?;
    let (mut solved, mut world, mut waypoints) = (vec![], vec![], vec![]);
    for (&p, path) in failed.iter().zip(found.paths) {
        if let Some(path) = path {
            solved.push(p);
            world.push(result.problems[p].world);
            waypoints.push(path);
        }
    }
    if solved.is_empty() {
        return Ok(());
    }
    shortcut_until(device, worlds, &world, &mut waypoints, &f.shortcut, deadline)?;
    let (n, points) = (result.paths.dof, result.paths.points);
    let mut refit = JointPaths::zeros(solved.len(), points, n);
    for (i, path) in waypoints.iter().enumerate() {
        if !trace(path, n, refit.path_mut(i)) {
            spread(path, n, refit.path_mut(i));
        }
    }
    let mut optimized = refit.clone();
    device.trajopt(worlds, &world, &mut optimized, o, deadline)?;
    // The optimized splines, then the refit ones.
    let candidates = JointPaths { positions: [optimized.positions, refit.positions].concat(), ..refit };
    let both_worlds: Vec<u32> = world.iter().chain(&world).copied().collect();
    let (clearance, lengths) = validate(device, worlds, &candidates, &both_worlds, o)?;
    for (i, &p) in solved.iter().enumerate() {
        let best = [i, solved.len() + i]
            .into_iter()
            .filter(|&c| clearance[c] >= 0.0)
            .min_by(|&a, &b| lengths[a].total_cmp(&lengths[b]));
        if let Some(c) = best {
            let item = p * result.seeds;
            result.paths.path_mut(item).copy_from_slice(candidates.path(c));
            result.valid[item] = true;
            result.min_clearance[item] = clearance[c];
            result.length[item] = lengths[c];
        }
    }
    Ok(())
}

/// B-spline control points whose curve runs exactly along a waypoint path: each waypoint three
/// times (the curve stops there), and the other points along the edges in proportion to their
/// lengths (the curve runs straight between waypoints). False if `out` has too few points.
fn trace(path: &[f32], n: usize, out: &mut [f32]) -> bool {
    let (waypoints, t_count) = (path.len() / n, out.len() / n);
    if 3 * waypoints > t_count {
        return false;
    }
    let edges: Vec<f32> = path.chunks(n).zip(path.chunks(n).skip(1)).map(|(a, b)| distance(a, b)).collect();
    let (extra, total) = (t_count - 3 * waypoints, edges.iter().sum::<f32>().max(1e-9));
    // Largest remainders, so the counts add up to `extra`.
    let share = |e: usize| extra as f32 * edges[e] / total;
    let mut counts: Vec<usize> = (0..edges.len()).map(|e| share(e) as usize).collect();
    while counts.iter().sum::<usize>() < extra {
        let e =
            (0..edges.len()).max_by(|&a, &b| (share(a) - counts[a] as f32).total_cmp(&(share(b) - counts[b] as f32)));
        counts[e.expect("a path has an edge")] += 1;
    }
    let mut t = 0;
    for (w, q) in path.chunks(n).enumerate() {
        for _ in 0..3 {
            out[t * n..(t + 1) * n].copy_from_slice(q);
            t += 1;
        }
        if let Some(&count) = counts.get(w) {
            let next = &path[(w + 1) * n..(w + 2) * n];
            for c in 1..=count {
                let u = c as f32 / (count + 1) as f32;
                (0..n).for_each(|j| out[(t * n) + j] = q[j] + (next[j] - q[j]) * u);
                t += 1;
            }
        }
    }
    true
}

/// B-spline control points along a waypoint path: three at each end, the free ones at even
/// distances along it, as `seed_path` spreads them along its seeds.
fn spread(path: &[f32], n: usize, out: &mut [f32]) {
    let t_count = out.len() / n;
    let total = length(path, n);
    for t in 3..t_count - 3 {
        let (_, q) = locate(path, n, total * (t - 2) as f32 / (t_count - 5) as f32);
        out[t * n..(t + 1) * n].copy_from_slice(&q);
    }
    let (start, goal) = (&path[..n], &path[path.len() - n..]);
    for t in 0..3 {
        out[t * n..(t + 1) * n].copy_from_slice(start);
        out[(t_count - 1 - t) * n..(t_count - t) * n].copy_from_slice(goal);
    }
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
