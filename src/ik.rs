//! Batched collision-aware inverse kinematics: damped least squares from many seeds per target,
//! with the collision gradient projected into the Jacobian null space.

use crate::error::{Result, ensure_input};

use crate::device::{CollisionWeights, Device, Worlds, collision_free};
use crate::rng::Rng;
use crate::types::{Pose, Solved};

#[derive(Clone, Copy, Debug)]
pub struct IkOptions {
    pub seeds: usize,
    pub iterations: u32,
    /// Damped-least-squares damping (lambda).
    pub damping: f32,
    /// Meters of position error equivalent to one radian of rotation error.
    pub rot_weight: f32,
    /// Largest joint change per iteration (radians).
    pub max_step: f32,
    /// Step along the null-space-projected collision gradient.
    pub collision_step: f32,
    pub collision: CollisionWeights,
    pub position_tolerance: f32,
    pub rotation_tolerance: f32,
    pub rng_seed: u64,
}

impl Default for IkOptions {
    fn default() -> Self {
        Self {
            seeds: 32,
            iterations: 60,
            damping: 0.02,
            rot_weight: 0.5,
            max_step: 0.3,
            collision_step: 0.05,
            collision: CollisionWeights { world: 100.0, self_collision: 100.0, margin: 0.01, self_margin: 0.005 },
            position_tolerance: 1e-3,
            rotation_tolerance: 1e-2,
            rng_seed: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IkProblem {
    pub world: u32,
    pub target: Pose,
}

/// Per-seed results for `problems`; items are problem-major (`item = problem * seeds + seed`).
#[derive(Clone, Debug)]
pub struct IkResult {
    pub problems: Vec<IkProblem>,
    pub dof: usize,
    pub seeds: usize,
    /// `[problems, seeds, dof]`.
    pub q: Vec<f32>,
    pub position_error: Vec<f32>,
    pub rotation_error: Vec<f32>,
    pub world_clearance: Vec<f32>,
    pub self_clearance: Vec<f32>,
    pub success: Vec<bool>,
}

impl IkResult {
    pub fn solution(&self, item: usize) -> &[f32] {
        &self.q[item * self.dof..(item + 1) * self.dof]
    }

    /// The successful seed with the most obstacle clearance.
    pub fn best(&self, problem: usize) -> Option<&[f32]> {
        (problem * self.seeds..(problem + 1) * self.seeds)
            .filter(|&i| self.success[i])
            .max_by(|&a, &b| self.world_clearance[a].total_cmp(&self.world_clearance[b]))
            .map(|i| self.solution(i))
    }

    /// Every problem with a successful seed, with its best configuration.
    pub fn solved(&self) -> impl Iterator<Item = Solved<'_, IkProblem>> {
        self.problems
            .iter()
            .enumerate()
            .filter_map(|(index, problem)| Some(Solved { index, problem, solution: self.best(index)? }))
    }
}

/// Seed 0 starts from the robot's default configuration, the rest uniformly within joint limits.
pub fn solve_ik(device: &Device, worlds: &Worlds, problems: &[IkProblem], o: &IkOptions) -> Result<IkResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(o.seeds > 0, "need at least one seed");
    let positive = |v: f32| v.is_finite() && v > 0.0;
    ensure_input!(
        positive(o.max_step) && positive(o.rot_weight) && o.damping.is_finite() && o.damping >= 0.0,
        "IK needs a positive max_step and rot_weight and a non-negative damping"
    );
    for (i, p) in problems.iter().enumerate() {
        let (position, rotation) = (p.target.position, p.target.rotation);
        ensure_input!(
            position.is_finite() && rotation.is_finite() && (rotation.length() - 1.0).abs() < 1e-3,
            "IK problem {i}: the target needs a finite position and a unit-quaternion rotation"
        );
    }
    let items = problems.len() * o.seeds;
    let mut rng = Rng::new(o.rng_seed);
    let mut q = Vec::with_capacity(items * n);
    let mut item_world = Vec::with_capacity(items);
    let mut targets = Vec::with_capacity(items);
    for p in problems {
        for s in 0..o.seeds {
            if s == 0 {
                q.extend_from_slice(&robot.default_q);
            } else {
                q.extend((0..n).map(|j| rng.range(robot.lower[j], robot.upper[j])));
            }
            item_world.push(p.world);
            targets.push(p.target);
        }
    }
    let err = device.ik(worlds, &item_world, &targets, &mut q, o)?;
    let clear = device.clearance(worlds, &item_world, &q)?;
    let success = (0..items)
        .map(|i| err[i][0] < o.position_tolerance && err[i][1] < o.rotation_tolerance && collision_free(clear[i]))
        .collect();
    Ok(IkResult {
        problems: problems.to_vec(),
        dof: n,
        seeds: o.seeds,
        q,
        position_error: err.iter().map(|e| e[0]).collect(),
        rotation_error: err.iter().map(|e| e[1]).collect(),
        world_clearance: clear.iter().map(|c| c[0]).collect(),
        self_clearance: clear.iter().map(|c| c[1]).collect(),
        success,
    })
}
