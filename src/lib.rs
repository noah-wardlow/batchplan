//! Batched, collision-aware motion generation on any GPU wgpu supports (Vulkan on AMD, NVIDIA and
//! Intel; Metal; DX12), with a CPU implementation that produces the same results.
//!
//! - **Batch-first.** Every query covers many seeds, problems and worlds at once; a [`Device`]
//!   checks each batch and runs it in parallel on a GPU or on all CPU cores.
//! - **Separate algorithms over shared types.** [`ik`], [`trajopt`], [`timing`], [`datagen`] and
//!   the exporters ([`npy`], `lerobot`) are independent modules that take a `Device` and exchange
//!   the [`types`]; there are no planner plugins or runtime configuration. Results keep the
//!   problems they answer (`solved()`).
//! - **Standalone.** No middleware. Robots load from URDF; collision spheres are fitted to their
//!   geometry or loaded from a committed collision-model file. Datasets export to LeRobot v3.0
//!   with the optional `lerobot` feature.
//!
//! ```no_run
//! use batchplan::*;
//! let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
//! let device = Device::gpu(&robot)?;
//! let worlds = vec![World::default()];
//! let goal = vec![0.5, -1.2, 1.0, -1.4, -1.5, 0.3];
//! let problems = vec![PlanProblem { world: 0, start: robot.default_q().to_vec(), goal }];
//! let result = plan(&device, &worlds, &problems, &PlanOptions::default())?;
//! println!("solved: {}", result.best(0).is_some());
//! # anyhow::Ok(())
//! ```

mod cpu;
pub mod datagen;
mod description;
pub mod device;
mod gpu;
pub mod ik;
#[cfg(feature = "lerobot")]
pub mod lerobot;
pub mod npy;
pub mod rng;
pub mod robot;
pub mod spheres;
pub mod timing;
pub mod trajopt;
pub mod types;
mod urdf;
pub mod world;

pub use device::{CollisionWeights, Device, Evaluation};
pub use ik::{IkOptions, IkProblem, IkResult, solve_ik};
pub use robot::{CollisionModel, Robot, RobotOptions};
pub use spheres::{SphereGeometry, SphereOptions};
pub use trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
pub use types::{JointPaths, JointTrajectory, Pose, Solved};
pub use world::{Obstacle, World};
