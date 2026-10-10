//! Where batched work runs. A [`Device`] owns an immutable copy of the robot (uploaded once on
//! the GPU) and evaluates independent items in parallel. Worlds are uploaded once as [`Worlds`],
//! and items reference them by index.
//!
//! The CPU and GPU implementations sit behind a crate-private `Backend` trait and are kept
//! numerically interchangeable, so code written against a `Device` runs unchanged on either.
//! Every batch is checked here before it reaches either one, so malformed input is an `Err` on
//! both rather than a panic on one and a plausible wrong answer on the other.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};

use crate::cpu::CpuBackend;
use crate::gpu::GpuBackend;
use crate::ik::IkOptions;
use crate::robot::Robot;
use crate::trajopt::PlanOptions;
use crate::types::{JointPaths, Pose};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct CollisionWeights {
    pub world: f32,
    pub self_collision: f32,
    /// Obstacle clearance below which the world cost activates (meters).
    pub margin: f32,
    /// Self clearance below which the self-collision cost activates (meters).
    pub self_margin: f32,
}

impl CollisionWeights {
    /// Zero weights: evaluate clearances only.
    pub const NONE: Self = Self { world: 0.0, self_collision: 0.0, margin: 0.0, self_margin: 0.0 };
}

/// Per-configuration collision evaluation, indexed by item. Clearances are 1e30 when there is
/// nothing to check (an empty world, or no self-collision pairs).
#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    /// Signed distance from the robot spheres to the nearest obstacle (negative = penetrating).
    pub world_clearance: Vec<f32>,
    /// Signed distance between the closest checked pair of robot spheres.
    pub self_clearance: Vec<f32>,
    pub cost: Vec<f32>,
    /// d(cost)/dq, `[items, dof]`.
    pub grad: Vec<f32>,
}

impl Evaluation {
    pub fn collision_free(&self, item: usize) -> bool {
        self.world_clearance[item] >= 0.0 && self.self_clearance[item] >= 0.0
    }
}

pub struct Device {
    id: u64,
    backend: Box<dyn Backend>,
}

/// Worlds prepared for one [`Device`] by [`Device::upload`]. On a GPU their obstacles and distance
/// grids stay in device memory, so every call that takes them reuses one upload.
pub struct Worlds {
    device: u64,
    worlds: Vec<World>,
    prepared: Box<dyn Any + Send + Sync>,
}

impl Worlds {
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    pub fn as_slice(&self) -> &[World] {
        &self.worlds
    }

    /// The backend's form of the worlds; `Device` checks they were prepared by that backend.
    pub(crate) fn prepared<T: 'static>(&self) -> &T {
        self.prepared.downcast_ref().expect("Device checks that worlds were uploaded to it")
    }
}

impl Device {
    /// All CPU cores (rayon).
    pub fn cpu(robot: &Robot) -> Self {
        Self::new(Box::new(CpuBackend::new(robot)))
    }

    /// The fastest GPU wgpu finds: Vulkan on Linux and Windows, Metal on macOS, DX12 as fallback.
    pub fn gpu(robot: &Robot) -> Result<Self> {
        Ok(Self::new(Box::new(GpuBackend::new(robot, None)?)))
    }

    /// The first adapter whose name contains `name` (case-insensitive), e.g. `"radv"` or `"llvmpipe"`.
    pub fn gpu_named(robot: &Robot, name: &str) -> Result<Self> {
        Ok(Self::new(Box::new(GpuBackend::new(robot, Some(name))?)))
    }

    fn new(backend: Box<dyn Backend>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), backend }
    }

    /// Prepares `worlds` for this device. Calls on other devices refuse the result.
    pub fn upload(&self, worlds: &[World]) -> Result<Worlds> {
        Ok(Worlds { device: self.id, worlds: worlds.to_vec(), prepared: self.backend.upload(worlds)? })
    }

    pub fn name(&self) -> String {
        self.backend.name()
    }

    pub fn robot(&self) -> &Robot {
        self.backend.robot()
    }

    /// Collision cost, its gradient and clearances for each configuration in `q` (`[items, dof]`)
    /// against `worlds[item_world[item]]`. Errors if `q` does not hold `dof` values per item, an
    /// item names a world outside `worlds`, or `worlds` were uploaded to another device.
    pub fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        self.backend.evaluate(worlds, item_world, q, w)
    }

    pub(crate) fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        ensure!(targets.len() == item_world.len(), "{} IK targets for {} items", targets.len(), item_world.len());
        self.backend.ik(worlds, item_world, targets, q, o)
    }

    pub(crate) fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
    ) -> Result<()> {
        ensure!(
            paths.dof == self.robot().dof(),
            "paths have {} joints, the robot has {}",
            paths.dof,
            self.robot().dof()
        );
        ensure!(paths.points >= 7, "paths need at least 7 control points");
        self.check_batch(worlds, item_world, paths.positions.len(), paths.points * paths.dof)?;
        self.backend.trajopt(worlds, item_world, paths, o)
    }

    /// The device, shape and index invariants every batch must satisfy before it reaches a backend.
    fn check_batch(&self, worlds: &Worlds, item_world: &[u32], values: usize, per_item: usize) -> Result<()> {
        ensure!(worlds.device == self.id, "these worlds were uploaded to a different device");
        let items = item_world.len();
        ensure!(
            values == items * per_item,
            "{items} items need {} values ({per_item} each), got {values}",
            items * per_item
        );
        if let Some((item, &world)) = item_world.iter().enumerate().find(|&(_, &w)| w as usize >= worlds.len()) {
            bail!("item {item} references world {world}, but only {} worlds were given", worlds.len());
        }
        Ok(())
    }
}

/// The parallel inner loops each device implements. Algorithm modules own seeding, validation
/// and selection so both implementations see identical inputs.
pub(crate) trait Backend: Send + Sync {
    fn name(&self) -> String;
    fn robot(&self) -> &Robot;
    /// The backend's own form of `worlds`, which [`Worlds::prepared`] hands back to it.
    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>>;
    fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation>;
    /// Runs IK in place on `q` (`[items, dof]`) toward `targets[item]`; returns `[position error, rotation error]` per item.
    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>>;
    /// Optimizes each path's control points in place, holding the three at each end fixed.
    fn trajopt(&self, worlds: &Worlds, item_world: &[u32], paths: &mut JointPaths, o: &PlanOptions) -> Result<()>;
}
