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
use crate::types::{JointPaths, Solved, StartMotion};

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
    /// Where the path ends, up to whole turns: continuous joints turn the short way (or the long
    /// way, if that is the one that is free), and a joint whose range spans more than a turn may
    /// end a turn away from this value when that is nearer or free. Every such end is the same pose.
    pub goal: Vec<f32>,
    /// How the robot is already moving at `start`, to replan mid-motion; `None` starts at rest.
    /// The path's first three control points then continue that motion, and
    /// [`crate::Trajectory::moving`] times it.
    pub start_motion: Option<StartMotion>,
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
        ensure_input!(
            robot.within(&p.start) && robot.within(&p.goal),
            "problem {pi}: start and goal must be within the joint limits"
        );
        if let Some(m) = &p.start_motion {
            ensure_input!(
                m.velocity.len() == n && m.acceleration.len() == n,
                "problem {pi}: the start motion needs {n} velocities and accelerations"
            );
            let within = |v: &[f32], limits: &[f32]| v.iter().zip(limits).all(|(x, l)| x.is_finite() && x.abs() <= *l);
            ensure_input!(
                within(&m.velocity, robot.max_velocity()) && within(&m.acceleration, robot.max_acceleration()),
                "problem {pi}: the start motion exceeds the velocity or acceleration limits"
            );
        }
        // Seed 0 runs straight to the nearest equivalent goal; odd seeds run straight to the others
        // while they last, the rest bend toward the nearest through a random via point.
        let goals = robot.goal_variants(&p.start, &p.goal);
        for s in 0..o.seeds {
            let (goal, straight) = match s {
                0 => (&goals[0], true),
                s if s % 2 == 1 && s.div_ceil(2) < goals.len() => (&goals[s.div_ceil(2)], true),
                _ => (&goals[0], false),
            };
            let path = paths.path_mut(pi * o.seeds + s);
            seed_path(robot, &p.start, goal, straight, &mut rng, path);
            if let Some(m) = &p.start_motion {
                continue_motion(robot, &p.start, goal, m, path);
            }
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
    let clear = device.clearance(worlds, &dense_world, &dense)?;
    let min_clearance = (0..items)
        .map(|item| {
            (item * samples..(item + 1) * samples)
                .map(|i| clear[i][0].min(clear[i][1]))
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
            let goals = result.problems[p].goal.clone();
            RrtProblem {
                world: problem.world,
                start: problem.start.clone(),
                goals: device.robot().goal_variants(&problem.start, &goals),
            }
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
        let problem = &result.problems[solved[i]];
        if let Some(m) = &problem.start_motion {
            let goal = &path[path.len() - n..];
            continue_motion(device.robot(), &problem.start, goal, m, refit.path_mut(i));
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

/// Pins `path`'s first three control points so the spline leaves `start` with `motion`'s velocity
/// and acceleration at the knot interval uniform timing gives a straight path to `goal` (see
/// [`crate::Trajectory::moving`]).
fn continue_motion(robot: &Robot, start: &[f32], goal: &[f32], motion: &StartMotion, path: &mut [f32]) {
    let n = start.len();
    let mut straight = vec![0.0; path.len()];
    for (t, q) in straight.chunks_mut(n).enumerate() {
        let u = (t.saturating_sub(2) as f32 / (path.len() / n - 5) as f32).min(1.0);
        q.iter_mut().enumerate().for_each(|(j, v)| *v = start[j] + (goal[j] - start[j]) * u);
    }
    // A quarter slower than the straight path allows, leaving room for the bends optimization adds;
    // a straight path that does not move still needs a time scale to leave its start with.
    let h0 = (1.25 * crate::timing::uniform_knot(robot, &straight)).max(1e-2);
    let (v, a) = (&motion.velocity, &motion.acceleration);
    for j in 0..n {
        path[j] = start[j] - v[j] * h0 + a[j] * h0 * h0 / 3.0;
        path[n + j] = start[j] - a[j] * h0 * h0 / 6.0;
        path[2 * n + j] = start[j] + v[j] * h0 + a[j] * h0 * h0 / 3.0;
    }
}

/// Control points on the straight line from `start` to `goal`, or bent through a random via point.
/// The first three and last three are the start and goal.
fn seed_path(robot: &Robot, start: &[f32], goal: &[f32], straight: bool, rng: &mut Rng, out: &mut [f32]) {
    let n = start.len();
    let t_count = out.len() / n;
    let (via_u, via): (f32, Vec<f32>) = if straight {
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
