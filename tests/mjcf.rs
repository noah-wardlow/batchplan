//! MJCF loading: MuJoCo Menagerie models against our URDFs, MuJoCo's conventions on a handwritten
//! model, and scenes as worlds.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::path::PathBuf;

use batchplan::rng::Rng;
use batchplan::*;
use glam::{DAffine3, DMat3, DQuat, DVec3, Vec3};

/// Kinematics only: these Menagerie models reference meshes that are not bundled.
fn no_spheres() -> RobotOptions {
    RobotOptions { collision_model: Some(CollisionModel::default()), ..Default::default() }
}

fn affine(p: Pose) -> DAffine3 {
    DAffine3::from_rotation_translation(p.rotation.as_dquat(), p.position.as_dvec3())
}

/// Largest difference between two transforms (rotation entries and translation in meters).
fn difference(a: DAffine3, b: DAffine3) -> f64 {
    let rot = (a.matrix3 - b.matrix3).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    rot.max((a.translation - b.translation).length())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("batchplan-mjcf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn menagerie_panda_matches_the_urdf_panda() {
    // Menagerie couples the fingers with a joint equality; locking one locks both.
    let options = RobotOptions { lock_joints: HashMap::from([("finger_joint2".into(), 0.04)]), ..no_spheres() };
    let mjcf = Robot::load(common::asset("menagerie/franka_emika_panda/panda.xml"), &options).unwrap();
    let urdf = common::panda().unwrap();
    assert_eq!(mjcf.joint_names(), ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]);
    assert_eq!(mjcf.lower(), urdf.lower());
    assert_eq!(mjcf.upper(), urdf.upper());
    let mut rng = Rng::new(5);
    let mut worst = 0.0f64;
    for _ in 0..100 {
        let q: Vec<f32> = (0..7).map(|j| rng.range(urdf.lower()[j], urdf.upper()[j])).collect();
        let pairs = (0..8)
            .map(|k| (format!("link{k}"), format!("panda_link{k}")))
            .chain([("hand".into(), "panda_hand".into())]);
        for (m, u) in pairs {
            worst =
                worst.max(difference(affine(mjcf.link_pose(&q, &m).unwrap()), affine(urdf.link_pose(&q, &u).unwrap())));
        }
        let (left, right) = (mjcf.link_pose(&q, "left_finger").unwrap(), mjcf.link_pose(&q, "right_finger").unwrap());
        assert!(((left.position - right.position).length() - 0.08).abs() < 1e-5, "fingers are not 4 cm open each");
    }
    assert!(worst < 1e-5, "link poses differ by {worst}");
}

#[test]
fn menagerie_ur5e_matches_the_urdf_ur5e_up_to_frame_offsets() {
    // Menagerie's world frame is the UR controller's base frame, which the ROS description keeps
    // as its `base` link (rotated half a turn about z). Body frames sit at different points of
    // each link than the URDF's, a constant offset per link, and Menagerie rounds a few
    // dimensions to the millimeter (0.163 m shoulder height against 0.1625 m).
    let mjcf =
        Robot::load(common::asset("menagerie/universal_robots_ur5e/ur5e.xml"), &RobotOptions::default()).unwrap();
    let urdf = Robot::load(common::asset("ur5e/ur_description/urdf/ur5e.urdf"), &RobotOptions::default()).unwrap();
    assert_eq!(mjcf.joint_names(), urdf.joint_names());
    let zero = vec![0.0; 6];
    let world = affine(urdf.link_pose(&zero, "base").unwrap()).inverse();
    let mut rng = Rng::new(6);
    let mut worst = 0.0f64;
    for link in ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"] {
        let offset = (world * affine(urdf.link_pose(&zero, link).unwrap())).inverse()
            * affine(mjcf.link_pose(&zero, link).unwrap());
        for _ in 0..50 {
            let q: Vec<f32> = (0..6).map(|_| rng.range(-3.0, 3.0)).collect();
            let expected = world * affine(urdf.link_pose(&q, link).unwrap()) * offset;
            worst = worst.max(difference(expected, affine(mjcf.link_pose(&q, link).unwrap())));
        }
    }
    assert!(worst < 3e-3, "links differ by {worst} beyond constant offsets");
    // Its collision geometry is primitives, so spheres fit without any mesh files.
    assert!(!mjcf.collision_model().spheres.is_empty());
}

#[test]
fn menagerie_2f85_couples_its_drivers() {
    let robot = Robot::load(common::asset("menagerie/robotiq_2f85/2f85.xml"), &no_spheres()).unwrap();
    // right_driver_joint follows left_driver_joint; the four-bar `connect` loops are not
    // representable in a tree, so the passive joints stay independent.
    assert!(!robot.joint_names().iter().any(|j| j == "right_driver_joint"));
    assert_eq!(robot.dof(), 7);
    let left = robot.joint_names().iter().position(|j| j == "left_driver_joint").unwrap();
    let mut q = robot.default_q().to_vec();
    let before = robot.link_pose(&q, "right_driver").unwrap();
    q[left] = 0.5;
    let after = robot.link_pose(&q, "right_driver").unwrap();
    assert!((before.rotation.angle_between(after.rotation) - 0.5).abs() < 1e-3, "the right driver does not follow");
}

const CONVENTIONS: &str = r#"<mujoco model="conventions">
  <include file="defaults.xml"/>
  <worldbody>
    <body name="base" pos="0 0 0.1" euler="10 20 30" childclass="arm">
      <joint name="j1" axis="0 0 1"/>
      <body name="upper" pos="0.2 0 0" axisangle="0 0 1 30">
        <joint name="j2"/>
        <joint name="j3" type="slide" axis="1 0 0" range="0 0.1" pos="0.05 0 0"/>
        <geom fromto="0 0 0 0.3 0 0"/>
        <body name="tip" pos="0.3 0 0" xyaxes="0 1 0 -1 0 0">
          <joint name="j4" pos="0 0 0.05" range="-45 45"/>
          <body name="tool" pos="0 0 0.1" zaxis="1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"#;

const DEFAULTS: &str = r#"<mujoco>
  <default>
    <default class="arm">
      <joint axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="0.02"/>
    </default>
  </default>
</mujoco>"#;

#[test]
fn mjcf_follows_mujoco_conventions() {
    let dir = scratch("conventions");
    std::fs::write(dir.join("model.xml"), CONVENTIONS).unwrap();
    std::fs::write(dir.join("defaults.xml"), DEFAULTS).unwrap();
    let robot = Robot::load(dir.join("model.xml"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.joint_names(), ["j1", "j2", "j3", "j4"]);
    // Angles are degrees unless the compiler says otherwise; slide ranges are meters.
    let (q, h) = (FRAC_PI_2 as f32, std::f32::consts::FRAC_PI_4);
    assert_eq!(robot.lower(), [-q, -q, 0.0, -h]);
    assert_eq!(robot.upper(), [q, q, 0.1, h]);
    let deg = |d: f64| d.to_radians();
    let t = |x: f64, y: f64, z: f64| DAffine3::from_translation(DVec3::new(x, y, z));
    let r = |m: DMat3| DAffine3::from_mat3(m);
    let mut rng = Rng::new(2);
    for _ in 0..50 {
        let q: Vec<f32> = (0..4).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        let [q1, q2, q3, q4] = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
        // Euler angles default to the intrinsic x-y-z sequence.
        let base_rot =
            DMat3::from_rotation_x(deg(10.0)) * DMat3::from_rotation_y(deg(20.0)) * DMat3::from_rotation_z(deg(30.0));
        let expected = t(0.0, 0.0, 0.1)
            * r(base_rot)
            * r(DMat3::from_rotation_z(q1))
            * t(0.2, 0.0, 0.0)
            * r(DMat3::from_rotation_z(deg(30.0)))
            * r(DMat3::from_rotation_y(q2))
            * t(q3, 0.0, 0.0)
            * t(0.3, 0.0, 0.0)
            * r(DMat3::from_rotation_z(FRAC_PI_2)) // xyaxes "0 1 0 -1 0 0"
            * t(0.0, 0.0, 0.05)
            * r(DMat3::from_rotation_y(q4))
            * t(0.0, 0.0, -0.05)
            * t(0.0, 0.0, 0.1)
            * r(DMat3::from_quat(DQuat::from_rotation_arc(DVec3::Z, DVec3::X))); // zaxis "1 0 0"
        let got = affine(robot.link_pose(&q, "tool").unwrap());
        assert!(difference(got, expected) < 1e-5, "tool pose off by {} at {q:?}", difference(got, expected));
    }
    // The fromto capsule spans the upper body: spheres were fitted along it.
    let spheres = &robot.collision_model().spheres["upper"];
    let reach = spheres.iter().map(|s| s[0] + s[3]).fold(f32::MIN, f32::max);
    assert!((reach - 0.32).abs() < 0.01, "capsule should reach x = 0.32, spheres reach {reach}");
}

#[test]
fn scenes_load_as_worlds_without_their_robots() {
    let floor = World::load(common::asset("menagerie/franka_emika_panda/scene.xml")).unwrap();
    assert_eq!(floor.obstacles.len(), 1, "the floor only; the Panda is a robot, not scene");
    let (top, _) = floor.obstacles[0].distance(Vec3::new(0.3, 0.2, 0.0));
    assert!(top.abs() < 1e-6, "the floor's surface is z = 0");

    let dir = scratch("scene");
    let scene = r#"<mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <geom name="post" type="cylinder" pos="0.5 0 0.3" size="0.05 0.3"/>
        <body name="table" pos="0.6 0 0">
          <geom type="box" size="0.3 0.4 0.02" pos="0 0 -0.02"/>
          <geom type="sphere" size="0.1" contype="0" conaffinity="0"/>
        </body>
        <body name="arm" pos="0 0 0">
          <joint name="j" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        </body>
      </worldbody>
    </mujoco>"#;
    std::fs::write(dir.join("scene.xml"), scene).unwrap();
    let world = World::load(dir.join("scene.xml")).unwrap();
    assert_eq!(world.obstacles.len(), 2, "the post and the table top; not the visual sphere or the arm");
    let robot = Robot::load(dir.join("scene.xml"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.joint_names(), ["j"]);
    assert_eq!(robot.collision_model().spheres.len(), 1, "only the arm's geometry belongs to the robot");
}
