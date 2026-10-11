//! Planner output as policy-training demonstrations: nominal reaches plus recoveries from
//! perturbed states, timed within the robot's limits at randomized speeds.

use crate::device::{Device, Worlds, collision_free};
use crate::error::{Result, ensure_input};
use crate::ik::{IkOptions, IkProblem, solve_ik};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::spline;
use crate::timing::Trajectory;
use crate::trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
use crate::types::{JointTrajectory, Pose, Solved};
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
    /// Index in `result.problems` of the problem whose best path was perturbed.
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
    worlds: &Worlds,
    result: &PlanResult,
    o: &RecoveryOptions,
) -> Result<Vec<Recovery>> {
    let robot = device.robot();
    let n = robot.dof();
    let mut rng = Rng::new(o.rng_seed);
    let mut candidates = vec![];
    let mut on_path = vec![0.0; n];
    for Solved { index: parent, problem, solution: path } in result.solved() {
        for _ in 0..o.per_trajectory {
            let phase = rng.range(o.phase.0, o.phase.1);
            spline::at_phase(path, n, phase, &mut on_path);
            let start: Vec<f32> = (0..n)
                .map(|j| (on_path[j] + o.sigma * rng.normal()).clamp(robot.bounds(j).0, robot.bounds(j).1))
                .collect();
            let next = PlanProblem { world: problem.world, start, goal: problem.goal.clone(), start_motion: None };
            candidates.push(Recovery { parent, phase, problem: next });
        }
    }
    let starts: Vec<f32> = candidates.iter().flat_map(|c| c.problem.start.iter().copied()).collect();
    let item_world: Vec<u32> = candidates.iter().map(|c| c.problem.world).collect();
    let clear = device.clearance(worlds, &item_world, &starts)?;
    Ok(candidates.into_iter().enumerate().filter(|&(i, _)| collision_free(clear[i])).map(|(_, c)| c).collect())
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

impl Origin {
    /// The demonstration a recovery branches from; `None` for nominal demonstrations.
    pub fn parent(&self) -> Option<usize> {
        match *self {
            Origin::Nominal => None,
            Origin::Recovery { parent, .. } => Some(parent),
        }
    }
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
    /// Each demonstration's speed scale (a fraction of the fastest timing within the robot's
    /// limits) is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
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
            dt: 0.05,
            rng_seed: 7,
        }
    }
}

/// Reach-to-pose demonstrations for each goal, followed by recoveries from perturbed states of
/// those demonstrations. Goals without a collision-free IK solution or plan are skipped.
pub fn demonstrations(
    device: &Device,
    worlds: &Worlds,
    goals: &[IkProblem],
    o: &DemoOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let n = robot.dof();
    let (slowest, fastest) = o.speed_scale;
    ensure_input!(
        slowest > 0.0 && slowest <= fastest && fastest <= 1.0,
        "speed scales must satisfy 0 < slowest <= fastest <= 1, got {:?}",
        o.speed_scale
    );
    ensure_input!(o.dt.is_finite() && o.dt > 0.0, "dt must be positive, got {}", o.dt);
    let mut rng = Rng::new(o.rng_seed);
    let ik = solve_ik(device, worlds, goals, &o.ik)?;

    let mut starts: Vec<f32> = (0..goals.len())
        .flat_map(|_| {
            (0..n)
                .map(|j| {
                    (robot.default_q[j] + o.start_noise * rng.normal()).clamp(robot.bounds(j).0, robot.bounds(j).1)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = goals.iter().map(|g| g.world).collect();
    let start_clear = device.clearance(worlds, &item_world, &starts)?;
    for g in (0..goals.len()).filter(|&g| !collision_free(start_clear[g])) {
        starts[g * n..(g + 1) * n].copy_from_slice(&robot.default_q);
    }
    // Nominal plan problem p reaches the target of IK problem goal_of[p].
    let (goal_of, problems): (Vec<usize>, Vec<PlanProblem>) = ik
        .solved()
        .map(|s| {
            let start = starts[s.index * n..(s.index + 1) * n].to_vec();
            (s.index, PlanProblem { world: s.problem.world, start, goal: s.solution.to_vec(), start_motion: None })
        })
        .unzip();
    let nominal = plan(device, worlds, &problems, &o.plan)?;
    let recoveries = recovery_problems(device, worlds, &nominal, &o.recovery)?;
    let recovery_plans: Vec<PlanProblem> = recoveries.iter().map(|r| r.problem.clone()).collect();
    let recovered = plan(device, worlds, &recovery_plans, &o.plan)?;

    let mut timed = |path: &[f32]| {
        let speed_scale = rng.range(o.speed_scale.0, o.speed_scale.1);
        Trajectory::new(robot, path, speed_scale)?.sample(1.0 / o.dt)
    };
    let mut demos = vec![];
    // Demonstration index of each solved nominal problem, for recovery parents.
    let mut demo_of = vec![usize::MAX; problems.len()];
    for s in nominal.solved() {
        demo_of[s.index] = demos.len();
        let goal = goals[goal_of[s.index]].target;
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: s.problem.world,
            goal,
            trajectory: timed(s.solution)?,
        });
    }
    for s in recovered.solved() {
        let rec = &recoveries[s.index];
        let parent = demo_of[rec.parent];
        demos.push(Demonstration {
            origin: Origin::Recovery { parent, phase: rec.phase },
            world: s.problem.world,
            goal: demos[parent].goal,
            trajectory: timed(s.solution)?,
        });
    }
    Ok(demos)
}

/// The shared sample period of demonstrations to export, after checking that each one belongs to
/// one of `worlds`, has the robot's joints, holds whole samples, and shares that period.
pub(crate) fn check_demos(robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<f32> {
    ensure_input!(!demos.is_empty(), "no demonstrations to export");
    let dt = demos[0].trajectory.dt;
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        ensure_input!((d.world as usize) < worlds.len(), "demonstration {i} is in world {}, past the end", d.world);
        ensure_input!(t.dof == robot.dof(), "demonstration {i} has {} joints, the robot has {}", t.dof, robot.dof());
        ensure_input!(
            !t.positions.is_empty() && t.positions.len() == t.velocities.len() && t.positions.len() % t.dof == 0,
            "demonstration {i} does not hold whole samples"
        );
        ensure_input!(t.dt == dt, "all demonstrations must share one dt");
    }
    Ok(dt)
}
