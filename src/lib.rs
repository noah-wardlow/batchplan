//! Batched, collision-aware motion generation on any GPU wgpu supports (Vulkan on AMD, NVIDIA and
//! Intel; Metal; DX12), with a CPU implementation that produces the same results.
//!
//! - **Batch-first.** Every query covers many seeds, problems and worlds at once; a [`Device`]
//!   runs them in parallel on a GPU or on all CPU cores.
//! - **Separate algorithms over shared types.** [`ik`], [`trajopt`], [`timing`] and [`datagen`]
//!   are independent modules that take a `Device` and exchange the [`types`]; there are no
//!   planner plugins or runtime configuration.
//! - **Standalone.** No middleware; robots load from URDF plus a collision-sphere config.
//!   Datasets export to LeRobot v3.0 with the optional `lerobot` feature.
//!
//! ```no_run
//! use batchplan::*;
//! let robot = Robot::from_config_file("assets/franka/panda.json")?;
//! let device = Device::gpu(&robot)?;
//! let worlds = vec![World::default()];
//! let goal = vec![0.5, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8];
//! let problems = vec![PlanProblem { world: 0, start: robot.default_q().to_vec(), goal }];
//! let result = plan(&device, &worlds, &problems, &PlanOptions::default())?;
//! println!("solved: {}", result.best(0).is_some());
//! # anyhow::Ok(())
//! ```

mod cpu;
pub mod datagen;
pub mod device;
mod gpu;
pub mod ik;
#[cfg(feature = "lerobot")]
pub mod lerobot;
pub mod npy;
pub mod rng;
pub mod robot;
pub mod timing;
pub mod trajopt;
pub mod types;
pub mod world;

pub use device::{CollisionWeights, Device, Evaluation};
pub use ik::{IkOptions, IkProblem, IkResult, solve_ik};
pub use robot::Robot;
pub use trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
pub use types::{JointPaths, JointTrajectory, Pose, Solved};
pub use world::{Obstacle, World};
