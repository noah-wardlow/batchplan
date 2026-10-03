//! Helpers for turning planner output into policy-training demonstrations.

use anyhow::Result;

use crate::device::{CollisionWeights, Device};
use crate::rng::Rng;
use crate::trajopt::{PlanProblem, PlanResult};
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
