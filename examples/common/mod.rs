//! Random tabletop worlds shared by the examples and tests.
#![allow(dead_code)]

use std::f32::consts::PI;

use std::collections::HashMap;

use batchplan::rng::Rng;
use batchplan::{CollisionModel, Obstacle, Pose, Robot, RobotOptions, World};
use glam::{Quat, Vec3};

pub fn asset(path: &str) -> String {
    format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR"))
}

/// The Franka Panda with its fingers held open, a 10 cm tool frame below the hand as the IK
/// frame, and hand-tuned collision spheres converted from cuRobo.
pub fn panda_options() -> RobotOptions {
    let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
    RobotOptions {
        ee_link: Some("ee_link".into()),
        lock_joints: HashMap::from(fingers),
        default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
        collision_model: Some(CollisionModel::load(asset("franka/panda_collision.json")).expect("Panda spheres")),
        ..Default::default()
    }
}

pub fn panda() -> anyhow::Result<Robot> {
    Robot::load(asset("franka/franka_panda.urdf"), &panda_options())
}

/// A table at z = 0 (the robot base height) with 2-6 random boxes in front of the robot.
pub fn tabletop(rng: &mut Rng) -> World {
    let mut obstacles = vec![Obstacle::Cuboid {
        center: Vec3::new(0.4, 0.0, -0.02),
        half_extents: Vec3::new(0.7, 0.8, 0.02),
        rotation: Quat::IDENTITY,
    }];
    let boxes = 2 + (rng.uniform() * 5.0) as usize;
    for _ in 0..boxes {
        let half = Vec3::new(rng.range(0.02, 0.08), rng.range(0.02, 0.08), rng.range(0.03, 0.2));
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.3, 0.75), rng.range(-0.5, 0.5), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, PI)),
        });
    }
    World { obstacles }
}

/// A top-down grasp pose (gripper z axis pointing down) in free space above the table.
pub fn grasp_target(world: &World, rng: &mut Rng) -> Pose {
    loop {
        let p = Vec3::new(rng.range(0.3, 0.7), rng.range(-0.45, 0.45), rng.range(0.05, 0.4));
        if world.obstacles.iter().all(|o| o.distance(p).0 > 0.06) {
            let rotation = Quat::from_rotation_z(rng.range(-PI, PI)) * Quat::from_rotation_x(PI);
            return Pose { position: p, rotation };
        }
    }
}
