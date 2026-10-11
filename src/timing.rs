//! Executable trajectories: planned B-spline paths timed so that position, velocity, acceleration
//! and jerk stay within the robot's limits.
//!
//! A trajectory is the planned path `P(s)`, a uniform cubic B-spline over `s` in spans, and a time
//! map `s = σ(t)`, a uniform cubic B-spline over time. Two timings are computed and the faster kept:
//! - Uniform: `σ(t) = t / h`. On the path, velocity is a quadratic B-spline over the control-point
//!   differences, acceleration a linear one over the second differences, and jerk is constant per
//!   span, so `h` chosen from those differences bounds the whole curve exactly.
//! - Time-optimal ([`crate::topp`]): the fastest velocity- and acceleration-limited timing of the
//!   same path, smoothed and stretched until velocity, acceleration and jerk, sampled densely
//!   along it, keep their limits.

use crate::error::{Error, Result, ensure_input};
use serde::{Deserialize, Serialize};

use crate::robot::Robot;
use crate::spline::{BASIS_D3, basis, basis_d1, basis_d2, blend};
use crate::topp::{self, CHECK_SAMPLES};
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

/// A rest-to-rest joint trajectory: a path whose first three and last three control points are
/// equal, and the time map that runs along it. Plain data, so it can be sent to another process;
/// [`Trajectory::check`] verifies one before it runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub dof: usize,
    /// The path, `[points, dof]`: a uniform cubic B-spline over `s` from 0 to `points - 3` spans.
    pub control_points: Vec<f32>,
    /// Seconds per span of the time map.
    pub knot_interval: f32,
    /// The time map's control points: a uniform cubic B-spline in `s`, from 0 at the start to the
    /// path's span count at the end.
    pub time_points: Vec<f32>,
}

/// The values of `d`-th differences of control points: (width, weights).
const DIFFERENCES: [(usize, &[f32]); 3] = [(2, &[-1.0, 1.0]), (3, &[1.0, -2.0, 1.0]), (4, &BASIS_D3)];

impl Trajectory {
    /// Times a planned path (`[points, dof]` control points, at least 7) as fast as the robot's
    /// velocity, acceleration and jerk limits allow, then slows it by `speed_scale` in (0, 1].
    pub fn new(robot: &Robot, control_points: &[f32], speed_scale: f32) -> Result<Self> {
        let n = robot.dof();
        ensure_input!(speed_scale > 0.0 && speed_scale <= 1.0, "speed_scale must be in (0, 1], got {speed_scale}");
        ensure_input!(
            control_points.len().is_multiple_of(n) && control_points.len() / n >= 7,
            "a path needs a whole number of at least 7 control points of {n} joints"
        );
        ensure_input!(control_points.iter().all(|v| v.is_finite()), "control points must be finite");
        let limits = [robot.max_velocity(), robot.max_acceleration(), robot.max_jerk()];
        let mut h = 0.0f32;
        for (order, &(width, weights)) in DIFFERENCES.iter().enumerate() {
            for (j, &limit) in limits[order].iter().enumerate() {
                let largest = largest_difference(control_points, n, j, width, weights);
                h = h.max((largest / limit).powf(1.0 / (order + 1) as f32));
            }
        }
        let spans = control_points.len() / n - 3;
        let (mut knot, mut points) = (h, linear(spans));
        if h > 0.0
            && let Some((tau, map)) = topp::time_map(robot, control_points)
            && tau * ((map.len() - 3) as f32) < h * spans as f32
        {
            (knot, points) = (tau, map);
        }
        Ok(Self {
            dof: n,
            control_points: control_points.to_vec(),
            knot_interval: knot / speed_scale,
            time_points: points,
        })
    }

    fn spans(&self) -> usize {
        self.control_points.len() / self.dof - 3
    }

    pub fn duration(&self) -> f32 {
        (self.time_points.len() - 3) as f32 * self.knot_interval
    }

    /// The state at time `t` (clamped to the trajectory), written into `out` without allocating.
    pub fn at(&self, t: f32, out: &mut JointState) {
        let n = self.dof;
        for v in [&mut out.position, &mut out.velocity, &mut out.acceleration] {
            v.resize(n, 0.0);
        }
        let (h, maps) = (self.knot_interval, self.time_points.len() - 3);
        // t / h at t = duration() can round to just under the last span's end.
        let x = if h <= 0.0 || t <= 0.0 {
            0.0
        } else if t >= self.duration() {
            maps as f32
        } else {
            (t / h).min(maps as f32)
        };
        let m = (x as usize).min(maps - 1);
        let u = x - m as f32;
        let tp = &self.time_points;
        let at = |w: [f32; 4]| (0..4).map(|i| w[i] * tp[m + i]).sum::<f32>();
        let (s, sd, sdd) = if h > 0.0 {
            (at(basis(u)), at(basis_d1(u)) / h, at(basis_d2(u)) / (h * h))
        } else {
            (at(basis(u)), 0.0, 0.0)
        };
        let spans = self.spans();
        let s = s.clamp(0.0, spans as f32);
        let span = (s as usize).min(spans - 1);
        let v = s - span as f32;
        let cp = &self.control_points;
        blend(cp, n, span, basis(v), &mut out.position);
        blend(cp, n, span, basis_d1(v), &mut out.velocity);
        blend(cp, n, span, basis_d2(v), &mut out.acceleration);
        for j in 0..n {
            let (p1, p2) = (out.velocity[j], out.acceleration[j]);
            out.velocity[j] = p1 * sd;
            out.acceleration[j] = p2 * sd * sd + p1 * sdd;
        }
        // At rest at an end, the position is exactly that end's control point (the blend of three
        // equal points can round off by one unit in the last place).
        let end = if x <= 0.0 {
            Some(0)
        } else if x >= maps as f32 {
            Some(cp.len() / n - 3)
        } else {
            None
        };
        if let Some(e) = end
            && (e..e + 3).all(|i| cp[i * n..(i + 1) * n] == cp[e * n..(e + 1) * n])
        {
            out.position.copy_from_slice(&cp[e * n..(e + 1) * n]);
        }
    }

    /// The trajectory sampled `hz` times per second, from its start to its exact end.
    pub fn sample(&self, hz: f32) -> Result<JointTrajectory> {
        let duration = self.duration();
        ensure_input!(
            hz.is_finite() && hz > 0.0 && (duration * hz) < 1e8,
            "cannot sample a {duration} s trajectory at {hz} Hz"
        );
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
        Ok(out)
    }

    /// Verifies the trajectory is safe to run on `robot`: finite, at rest at both ends, within every
    /// position limit, and within every velocity, acceleration and jerk limit: exactly for uniform
    /// timing, at 64 points per time-map span otherwise.
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
        let tp = &self.time_points;
        ensure_safe!(
            tp.len() >= 4 && tp.iter().all(|v| v.is_finite()),
            "the time map needs at least 4 finite control points"
        );
        ensure_safe!(tp.windows(2).all(|w| w[1] >= w[0]), "the time map must not run backward");
        let ends =
            [(tp[0] + 4.0 * tp[1] + tp[2]) / 6.0, (tp[tp.len() - 3] + 4.0 * tp[tp.len() - 2] + tp[tp.len() - 1]) / 6.0];
        let spans = (points - 3) as f32;
        ensure_safe!(
            ends[0].abs() < 1e-4 && (ends[1] - spans).abs() < 1e-4 * spans.max(1.0),
            "the time map must run from the path's start ({}) to its end ({})",
            ends[0],
            ends[1]
        );
        for end in [0, points - 3] {
            ensure_safe!(
                (end..end + 3).all(|i| cp[i * n..(i + 1) * n] == cp[end * n..(end + 1) * n]),
                "trajectory must start and end at rest (three equal control points at each end)"
            );
        }
        let limits = [robot.max_velocity(), robot.max_acceleration(), robot.max_jerk()];
        let names = ["velocity", "acceleration", "jerk"];
        let uniform = *tp == linear(points - 3);
        if !uniform && h > 0.0 {
            let mut out = vec![[0.0f32; 3]; n];
            for m in 0..tp.len() - 3 {
                for sample in 0..=CHECK_SAMPLES {
                    topp::derivatives(cp, n, h, tp, m, sample as f32 / CHECK_SAMPLES as f32, &mut out);
                    for (j, d) in out.iter().enumerate() {
                        for order in 0..3 {
                            let (peak, limit) = (d[order].abs(), limits[order][j]);
                            // False for NaN too.
                            let within = peak <= limit * (1.0 + 1e-3);
                            ensure_safe!(within, "joint {j} reaches {} {peak}, over its limit {limit}", names[order]);
                        }
                    }
                }
            }
        }
        for j in 0..n {
            let (lo, hi) = robot.bounds(j);
            if let Some(i) = (0..points).find(|&i| !(lo..=hi).contains(&cp[i * n + j])) {
                return Err(Error::Unsafe(format!("joint {j} leaves its range [{lo}, {hi}] at control point {i}")));
            }
            for (order, &(width, weights)) in DIFFERENCES.iter().enumerate().filter(|_| h > 0.0 && uniform) {
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

/// The time map of uniform timing over `spans` path spans: `σ(t) = t / h`.
fn linear(spans: usize) -> Vec<f32> {
    (0..spans + 3).map(|k| k as f32 - 1.0).collect()
}

/// Largest `|Σ weights[i] * cp[k + i]|` over every window of joint `j`.
fn largest_difference(cp: &[f32], n: usize, j: usize, width: usize, weights: &[f32]) -> f32 {
    let points = cp.len() / n;
    (0..points + 1 - width)
        .map(|k| weights.iter().enumerate().map(|(i, w)| w * cp[(k + i) * n + j]).sum::<f32>().abs())
        .fold(0.0, f32::max)
}
