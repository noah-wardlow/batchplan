//! Time parameterization of joint-space polylines.

use crate::robot::Robot;
use crate::types::JointTrajectory;

#[derive(Clone, Copy, Debug)]
pub struct RetimeOptions {
    /// Per-joint acceleration limit (rad/s^2); velocity limits come from the robot.
    pub max_acceleration: f32,
    /// In (0, 1]; scales the fastest feasible timing down (lower is slower).
    pub speed_scale: f32,
    /// Sample period of the output (seconds).
    pub dt: f32,
}

impl Default for RetimeOptions {
    fn default() -> Self {
        Self { max_acceleration: 5.0, speed_scale: 1.0, dt: 0.05 }
    }
}

/// Peak of ds/dtau and d2s/dtau2 for the minimum-jerk profile s = 10t^3 - 15t^4 + 6t^5.
const MIN_JERK_PEAK_VEL: f32 = 1.875;
const MIN_JERK_PEAK_ACC: f32 = 5.773_503;

/// Retimes the polyline through `path` (`[waypoints, dof]`) with a minimum-jerk (bell-shaped
/// velocity) profile along its joint-space arc length, the speed profile of human reaching.
///
/// The duration is the shortest that keeps every joint under the robot's velocity limits and
/// `o.max_acceleration` along straight segments, divided by `o.speed_scale`. Accelerations from
/// direction changes at waypoints are not bounded.
pub fn retime(robot: &Robot, path: &[f32], o: &RetimeOptions) -> JointTrajectory {
    let dof = robot.dof();
    let wp = |k: usize, j: usize| path[k * dof + j];
    let seg_len: Vec<f32> = (0..path.len() / dof - 1)
        .map(|k| (0..dof).map(|j| (wp(k + 1, j) - wp(k, j)).powi(2)).sum::<f32>().sqrt())
        .collect();
    let total: f32 = seg_len.iter().sum();
    if total < 1e-9 {
        return JointTrajectory {
            dof,
            dt: o.dt,
            duration: 0.0,
            positions: path[..dof].to_vec(),
            velocities: vec![0.0; dof],
        };
    }
    // Largest |dq_j/du| (u = arc length) per joint over all segments.
    let mut max_dir = vec![0.0f32; dof];
    for (k, &len) in seg_len.iter().enumerate() {
        if len > 1e-9 {
            for (j, m) in max_dir.iter_mut().enumerate() {
                *m = m.max(((wp(k + 1, j) - wp(k, j)) / len).abs());
            }
        }
    }
    let duration = (0..dof)
        .map(|j| {
            let by_vel = MIN_JERK_PEAK_VEL * total * max_dir[j] / robot.max_velocity()[j];
            let by_acc = (MIN_JERK_PEAK_ACC * total * max_dir[j] / o.max_acceleration).sqrt();
            by_vel.max(by_acc)
        })
        .fold(0.0f32, f32::max)
        / o.speed_scale;

    let samples = (duration / o.dt).ceil() as usize + 1;
    let mut positions = Vec::with_capacity(samples * dof);
    let mut velocities = Vec::with_capacity(samples * dof);
    for h in 0..samples {
        let tau = (h as f32 * o.dt / duration).min(1.0);
        let s = tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau));
        let s_dot = 30.0 * tau * tau * (1.0 - tau) * (1.0 - tau) / duration;
        let u = s * total;
        // Locate the segment containing arc length u.
        let mut k = 0;
        let mut start = 0.0;
        while k < seg_len.len() - 1 && start + seg_len[k] < u {
            start += seg_len[k];
            k += 1;
        }
        let a = if seg_len[k] > 1e-9 { ((u - start) / seg_len[k]).clamp(0.0, 1.0) } else { 0.0 };
        for j in 0..dof {
            let d = wp(k + 1, j) - wp(k, j);
            positions.push(wp(k, j) + a * d);
            velocities.push(if seg_len[k] > 1e-9 { d / seg_len[k] * total * s_dot } else { 0.0 });
        }
    }
    JointTrajectory { dof, dt: o.dt, duration, positions, velocities }
}
