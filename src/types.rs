//! Standard data types. Every algorithm takes and returns these (or plain row-major arrays with
//! the documented shapes) instead of hiding behind plugin interfaces.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
}

/// A problem that has a solution: its position in the batch, the problem itself and its best
/// solution (a configuration for IK, a `[waypoints, dof]` path for planning).
#[derive(Clone, Copy, Debug)]
pub struct Solved<'a, P> {
    pub index: usize,
    pub problem: &'a P,
    pub solution: &'a [f32],
}

/// A batch of joint-space paths with a shared waypoint count, row-major `[len, waypoints, dof]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPaths {
    pub dof: usize,
    pub waypoints: usize,
    pub positions: Vec<f32>,
}

impl JointPaths {
    pub fn zeros(len: usize, waypoints: usize, dof: usize) -> Self {
        Self { dof, waypoints, positions: vec![0.0; len * waypoints * dof] }
    }

    pub fn len(&self) -> usize {
        self.positions.len() / (self.waypoints * self.dof)
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Path `i` as `[waypoints, dof]`.
    pub fn path(&self, i: usize) -> &[f32] {
        let n = self.waypoints * self.dof;
        &self.positions[i * n..(i + 1) * n]
    }

    pub fn path_mut(&mut self, i: usize) -> &mut [f32] {
        let n = self.waypoints * self.dof;
        &mut self.positions[i * n..(i + 1) * n]
    }
}

/// One time-parameterized trajectory sampled every `dt` seconds, row-major `[len, dof]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTrajectory {
    pub dof: usize,
    pub dt: f32,
    pub duration: f32,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
}

impl JointTrajectory {
    pub fn len(&self) -> usize {
        self.positions.len() / self.dof
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}
