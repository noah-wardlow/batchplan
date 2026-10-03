//! Planner output as policy-training demonstrations: nominal reaches plus recoveries from
//! perturbed states, retimed with human-like speed profiles.

use anyhow::Result;

use crate::device::{CollisionWeights, Device};
use crate::ik::{IkOptions, IkProblem, solve_ik};
use crate::rng::Rng;
use crate::timing::{RetimeOptions, retime};
use crate::trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
use crate::types::{JointTrajectory, Pose};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct RecoveryOptions {
    pub per_trajectory: usize,
    /// Standard deviation of the joint-space perturbation (radians).
    pub sigma: f32,
    /// Range of path fractions where perturbations happen.
    pub phase: (f32, f32),
    pub rng_seed: u64,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self { per_trajectory: 2, sigma: 0.15, phase: (0.2, 0.8), rng_seed: 3 }
    }
}

/// A planning problem that starts from a perturbed state of a planned path.
#[derive(Clone, Debug)]
pub struct Recovery {
    /// Index of the nominal problem whose best path was perturbed.
    pub parent: usize,
    /// Fraction of the nominal path where the perturbation happened.
    pub phase: f32,
    pub problem: PlanProblem,
}

/// Simulates policy mistakes: perturbs states along the best path of each solved problem and
/// keeps the collision-free ones as problems toward the same goal. Planning them yields
/// demonstrations of recovering from off-nominal states, which raw planner output never contains.
pub fn recovery_problems(
    device: &Device,
    worlds: &[World],
    problems: &[PlanProblem],
    result: &PlanResult,
    o: &RecoveryOptions,
) -> Result<Vec<Recovery>> {
    let robot = device.robot();
    let n = robot.dof();
    let last = result.paths.waypoints - 1;
    let mut rng = Rng::new(o.rng_seed);
    let mut candidates = vec![];
    for (parent, problem) in problems.iter().enumerate() {
        let Some(path) = result.best(parent) else { continue };
        for _ in 0..o.per_trajectory {
            let phase = rng.range(o.phase.0, o.phase.1);
            let x = phase * last as f32;
            let k = (x as usize).min(last - 1);
            let a = x - k as f32;
            let start: Vec<f32> = (0..n)
                .map(|j| {
                    let on_path = path[k * n + j] + a * (path[(k + 1) * n + j] - path[k * n + j]);
                    (on_path + o.sigma * rng.normal()).clamp(robot.lower[j], robot.upper[j])
                })
                .collect();
            let next = PlanProblem { world: problem.world, start, goal: problem.goal.clone() };
            candidates.push(Recovery { parent, phase, problem: next });
        }
    }
    let starts: Vec<f32> = candidates.iter().flat_map(|c| c.problem.start.iter().copied()).collect();
    let item_world: Vec<u32> = candidates.iter().map(|c| c.problem.world).collect();
    let eval = device.evaluate(worlds, &item_world, &starts, &CollisionWeights::NONE)?;
    Ok(candidates.into_iter().enumerate().filter(|&(i, _)| eval.collision_free(i)).map(|(_, c)| c).collect())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Origin {
    Nominal,
    /// Replanned from a perturbed state of demonstration `parent` (an index into the same output),
    /// `phase` of the way along it.
    Recovery {
        parent: usize,
        phase: f32,
    },
}

/// One time-parameterized reach toward `goal` in `worlds[world]`.
#[derive(Clone, Debug)]
pub struct Demonstration {
    pub origin: Origin,
    pub world: u32,
    pub goal: Pose,
    pub trajectory: JointTrajectory,
}

#[derive(Clone, Copy, Debug)]
pub struct DemoOptions {
    pub ik: IkOptions,
    pub plan: PlanOptions,
    pub recovery: RecoveryOptions,
    /// Standard deviation of the joint noise added to the default pose for start states (radians).
    /// Starts that collide fall back to the default pose.
    pub start_noise: f32,
    /// Each demonstration's speed scale is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    pub max_acceleration: f32,
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    pub rng_seed: u64,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            recovery: RecoveryOptions::default(),
            start_noise: 0.3,
            speed_scale: (0.6, 1.0),
            max_acceleration: RetimeOptions::default().max_acceleration,
            dt: RetimeOptions::default().dt,
            rng_seed: 7,
        }
    }
}

/// Reach-to-pose demonstrations for each goal, followed by recoveries from perturbed states of
/// those demonstrations. Goals without a collision-free IK solution or plan are skipped.
pub fn demonstrations(
    device: &Device,
    worlds: &[World],
    goals: &[IkProblem],
    o: &DemoOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let n = robot.dof();
    let mut rng = Rng::new(o.rng_seed);
    let ik = solve_ik(device, worlds, goals, &o.ik)?;

    let mut starts: Vec<f32> = (0..goals.len())
        .flat_map(|_| {
            (0..n)
                .map(|j| (robot.default_q[j] + o.start_noise * rng.normal()).clamp(robot.lower[j], robot.upper[j]))
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = goals.iter().map(|g| g.world).collect();
    let start_eval = device.evaluate(worlds, &item_world, &starts, &CollisionWeights::NONE)?;
    for g in (0..goals.len()).filter(|&g| !start_eval.collision_free(g)) {
        starts[g * n..(g + 1) * n].copy_from_slice(&robot.default_q);
    }
    let (goal_of_problem, problems): (Vec<usize>, Vec<PlanProblem>) = (0..goals.len())
        .filter_map(|g| {
            let goal = ik.best(g)?.to_vec();
            Some((g, PlanProblem { world: goals[g].world, start: starts[g * n..(g + 1) * n].to_vec(), goal }))
        })
        .unzip();
    let nominal = plan(device, worlds, &problems, &o.plan)?;
    let recoveries = recovery_problems(device, worlds, &problems, &nominal, &o.recovery)?;
    let recovery_plans: Vec<PlanProblem> = recoveries.iter().map(|r| r.problem.clone()).collect();
    let recovered = plan(device, worlds, &recovery_plans, &o.plan)?;

    let mut retimed = |path: &[f32]| {
        let speed_scale = rng.range(o.speed_scale.0, o.speed_scale.1);
        retime(robot, path, &RetimeOptions { max_acceleration: o.max_acceleration, speed_scale, dt: o.dt })
    };
    let mut demos = vec![];
    let mut demo_of_problem = vec![None; problems.len()];
    for (p, problem) in problems.iter().enumerate() {
        if let Some(path) = nominal.best(p) {
            demo_of_problem[p] = Some(demos.len());
            let goal = goals[goal_of_problem[p]].target;
            demos.push(Demonstration {
                origin: Origin::Nominal,
                world: problem.world,
                goal,
                trajectory: retimed(path),
            });
        }
    }
    for (r, rec) in recoveries.iter().enumerate() {
        if let (Some(path), Some(parent)) = (recovered.best(r), demo_of_problem[rec.parent]) {
            demos.push(Demonstration {
                origin: Origin::Recovery { parent, phase: rec.phase },
                world: rec.problem.world,
                goal: demos[parent].goal,
                trajectory: retimed(path),
            });
        }
    }
    Ok(demos)
}
