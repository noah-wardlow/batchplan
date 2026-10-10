//! Executable trajectories: planned B-spline paths timed so that position, velocity, acceleration
//! and jerk stay within the robot's limits everywhere, not just at samples.
//!
//! On a uniform cubic B-spline with knot interval `h`, velocity is a quadratic B-spline over the
//! control-point differences divided by `h`, acceleration a linear one over the second differences
//! divided by `h²`, and jerk is constant per span (third differences over `h³`). Each therefore
//! stays within the largest of its control values, so choosing `h` from the differences bounds the
//! whole curve.

use crate::error::{Error, Result, ensure_input};
use serde::{Deserialize, Serialize};

use crate::robot::Robot;
use crate::spline::{BASIS_D3, basis, basis_d1, basis_d2, blend};
use crate::types::JointTrajectory;

/// Joint position, velocity and acceleration at one instant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointState {
    pub position: Vec<f32>,
    pub velocity: Vec<f32>,
    pub acceleration: Vec<f32>,
}

impl JointState {
    pub fn new(dof: usize) -> Self {
        Self { position: vec![0.0; dof], velocity: vec![0.0; dof], acceleration: vec![0.0; dof] }
    }
}

/// A rest-to-rest joint trajectory: a uniform cubic B-spline whose first three and last three
/// control points are equal, with `knot_interval` seconds per span. Plain data, so it can be sent
/// to another process; [`Trajectory::check`] verifies one before it runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub dof: usize,
    pub knot_interval: f32,
    /// `[points, dof]`.
    pub control_points: Vec<f32>,
}

/// The values of `d`-th differences of control points: (width, weights).
const DIFFERENCES: [(usize, &[f32]); 3] = [(2, &[-1.0, 1.0]), (3, &[1.0, -2.0, 1.0]), (4, &BASIS_D3)];

impl Trajectory {
    /// Times a planned path (`[points, dof]` control points) as fast as the robot's velocity,
    /// acceleration and jerk limits allow, then slows it by `speed_scale` in (0, 1].
    pub fn new(robot: &Robot, control_points: &[f32], speed_scale: f32) -> Self {
        let n = robot.dof();
        let limits = [robot.max_velocity(), robot.max_acceleration(), robot.max_jerk()];
        let mut h = 0.0f32;
        for (order, &(width, weights)) in DIFFERENCES.iter().enumerate() {
            for (j, &limit) in limits[order].iter().enumerate() {
                let largest = largest_difference(control_points, n, j, width, weights);
                h = h.max((largest / limit).powf(1.0 / (order + 1) as f32));
            }
        }
        Self { dof: n, knot_interval: h / speed_scale, control_points: control_points.to_vec() }
    }

    fn spans(&self) -> usize {
        self.control_points.len() / self.dof - 3
    }

    pub fn duration(&self) -> f32 {
        self.spans() as f32 * self.knot_interval
    }

    /// The state at time `t` (clamped to the trajectory), written into `out` without allocating.
    pub fn at(&self, t: f32, out: &mut JointState) {
        let n = self.dof;
        for v in [&mut out.position, &mut out.velocity, &mut out.acceleration] {
            v.resize(n, 0.0);
        }
        let cp = &self.control_points;
        if t <= 0.0 || t >= self.duration() {
            let end = if t <= 0.0 { &cp[..n] } else { &cp[cp.len() - n..] };
            out.position.copy_from_slice(end);
            out.velocity.fill(0.0);
            out.acceleration.fill(0.0);
            return;
        }
        let h = self.knot_interval;
        let x = t / h;
        let span = (x as usize).min(self.spans() - 1);
        let u = x - span as f32;
        let scaled = |w: [f32; 4], s: f32| w.map(|v| v * s);
        blend(cp, n, span, basis(u), &mut out.position);
        blend(cp, n, span, scaled(basis_d1(u), 1.0 / h), &mut out.velocity);
        blend(cp, n, span, scaled(basis_d2(u), 1.0 / (h * h)), &mut out.acceleration);
    }

    /// The trajectory sampled `hz` times per second, from its start to its exact end.
    pub fn sample(&self, hz: f32) -> JointTrajectory {
        let duration = self.duration();
        let dt = 1.0 / hz;
        let samples = (duration * hz).ceil() as usize + 1;
        let mut state = JointState::new(self.dof);
        let mut out = JointTrajectory {
            dof: self.dof,
            dt,
            duration,
            positions: Vec::with_capacity(samples * self.dof),
            velocities: Vec::with_capacity(samples * self.dof),
            accelerations: Vec::with_capacity(samples * self.dof),
        };
        for k in 0..samples {
            // The last sample lands exactly on the end, whatever the rounding of k * dt.
            self.at(if k + 1 == samples { duration } else { (k as f32 * dt).min(duration) }, &mut state);
            out.positions.extend_from_slice(&state.position);
            out.velocities.extend_from_slice(&state.velocity);
            out.accelerations.extend_from_slice(&state.acceleration);
        }
        out
    }

    /// Verifies the trajectory is safe to run on `robot`: finite, at rest at both ends, and within
    /// every position, velocity, acceleration and jerk limit along its whole length.
    pub fn check(&self, robot: &Robot) -> Result<()> {
        let n = robot.dof();
        let cp = &self.control_points;
        ensure_input!(self.dof == n, "trajectory has {} joints, the robot has {n}", self.dof);
        ensure_input!(
            cp.len().is_multiple_of(n) && cp.len() / n >= 7,
            "need a whole number of at least 7 control points"
        );
        macro_rules! ensure_safe {
            ($cond:expr, $($message:tt)+) => {
                if !$cond {
                    return Err(Error::Unsafe(format!($($message)+)));
                }
            };
        }
        ensure_safe!(cp.iter().all(|v| v.is_finite()), "control points must be finite");
        let h = self.knot_interval;
        ensure_safe!(h.is_finite() && h >= 0.0, "knot interval must be finite and non-negative, got {h}");
        let points = cp.len() / n;
        if h == 0.0 {
            ensure_safe!(cp.chunks(n).all(|p| p == &cp[..n]), "a trajectory that takes no time must not move");
        }
        for end in [0, points - 3] {
            ensure_safe!(
                (end..end + 3).all(|i| cp[i * n..(i + 1) * n] == cp[end * n..(end + 1) * n]),
                "trajectory must start and end at rest (three equal control points at each end)"
            );
        }
        let limits = [robot.max_velocity(), robot.max_acceleration(), robot.max_jerk()];
        let names = ["velocity", "acceleration", "jerk"];
        for j in 0..n {
            let (lo, hi) = (robot.lower()[j], robot.upper()[j]);
            if let Some(i) = (0..points).find(|&i| !(lo..=hi).contains(&cp[i * n + j])) {
                return Err(Error::Unsafe(format!("joint {j} leaves its range [{lo}, {hi}] at control point {i}")));
            }
            for (order, &(width, weights)) in DIFFERENCES.iter().enumerate().filter(|_| h > 0.0) {
                let peak = largest_difference(cp, n, j, width, weights) / h.powi(order as i32 + 1);
                let limit = limits[order][j];
                if peak > limit * (1.0 + 1e-4) {
                    return Err(Error::Unsafe(format!(
                        "joint {j} reaches {} {peak}, over its limit {limit}",
                        names[order]
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Largest `|Σ weights[i] * cp[k + i]|` over every window of joint `j`.
fn largest_difference(cp: &[f32], n: usize, j: usize, width: usize, weights: &[f32]) -> f32 {
    let points = cp.len() / n;
    (0..points + 1 - width)
        .map(|k| weights.iter().enumerate().map(|(i, w)| w * cp[(k + i) * n + j]).sum::<f32>().abs())
        .fold(0.0, f32::max)
}
