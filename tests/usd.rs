//! OpenUSD loading (feature `usd`): UsdPhysics conventions on handwritten fixtures, newton-assets
//! robots against the MJCF they were converted from, and scenes as worlds.
#![cfg(feature = "usd")]

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::*;
use glam::{DAffine3, DMat3, DVec3, Vec3};

fn affine(p: Pose) -> DAffine3 {
    DAffine3::from_rotation_translation(p.rotation.as_dquat(), p.position.as_dvec3())
}

fn difference(a: DAffine3, b: DAffine3) -> f64 {
    let rot = (a.matrix3 - b.matrix3).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    rot.max((a.translation - b.translation).length())
}

/// A pose authored in a Y-up, centimeter stage, as a Z-up pose in meters.
fn from_stage(stage: DAffine3) -> DAffine3 {
    let up = DMat3::from_rotation_x(std::f64::consts::FRAC_PI_2);
    DAffine3 { matrix3: up * stage.matrix3, translation: up * stage.translation * 0.01 }
}

#[test]
fn usd_follows_physics_conventions() {
    let t = |x: f64, y: f64, z: f64| DAffine3::from_translation(DVec3::new(x, y, z));
    let ry = |a: f64| DAffine3::from_rotation_y(a);
    let rz = |a: f64| DAffine3::from_rotation_z(a);
    for (variants, reach) in [(vec![], 25.0), (vec![("reach".to_string(), "short".to_string())], 20.0)] {
        let options = RobotOptions { variants, ..Default::default() };
        let robot = Robot::load(common::asset("usd/arm.usda"), &options).unwrap();
        // Finger and thumb mimic the shoulder, so they are not planning joints.
        assert_eq!(robot.joint_names(), ["shoulder", "elbow"]);
        let right = std::f32::consts::FRAC_PI_2;
        assert_eq!(robot.lower(), [-right, 0.0]);
        assert!((robot.upper()[0] - right).abs() < 1e-6 && (robot.upper()[1] - 0.05).abs() < 1e-7);
        let mut rng = Rng::new(9);
        for _ in 0..50 {
            let q: Vec<f32> = (0..2).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
            let (s, e) = (q[0] as f64, q[1] as f64 * 100.0);
            let base = t(0.0, 10.0, 0.0);
            // localPos1 = (-10, 0, 0): the body sits 10 cm from the joint along its x axis.
            let upper = base * t(0.0, 5.0, 0.0) * ry(s) * t(10.0, 0.0, 0.0);
            // Authored from forearm (body0) to upper (body1): the forearm slides back as it opens.
            let forearm = upper * t(reach - e, 0.0, 0.0);
            // NewtonMimicAPI: finger = 10 deg + 2 shoulder. PhysxMimicJointAPI with gearing -1: thumb = shoulder.
            let finger = forearm * t(5.0, 0.0, 0.0) * rz(10f64.to_radians() + 2.0 * s);
            let thumb = forearm * t(5.0, 0.0, 2.0) * rz(s);
            for (link, expected) in
                [("base", base), ("upper", upper), ("forearm", forearm), ("finger", finger), ("thumb", thumb)]
            {
                let got = affine(robot.link_pose(&q, link).unwrap());
                let err = difference(got, from_stage(expected));
                assert!(err < 1e-5, "{link} off by {err} at q = {q:?} (reach {reach})");
            }
        }
        // The instanced collider under the forearm counts as its geometry.
        let spheres = &robot.collision_model().spheres;
        for link in ["base", "upper", "forearm"] {
            assert!(spheres.contains_key(link), "no spheres on {link}");
        }
    }
}

#[test]
fn newton_ur5e_matches_the_mjcf_it_was_converted_from() {
    let usd =
        Robot::load(common::asset("newton/universal_robots_ur5e/usd_structured/ur5e.usda"), &RobotOptions::default())
            .unwrap();
    let mjcf =
        Robot::load(common::asset("menagerie/universal_robots_ur5e/ur5e.xml"), &RobotOptions::default()).unwrap();
    assert_eq!(usd.joint_names(), mjcf.joint_names());
    // USD authors these limits as +-360.00027 degrees; MJCF as +-6.28319 rad.
    for j in 0..6 {
        assert!((usd.lower()[j] - mjcf.lower()[j]).abs() < 1e-4 && (usd.upper()[j] - mjcf.upper()[j]).abs() < 1e-4);
    }
    let mut rng = Rng::new(4);
    let mut worst = 0.0f64;
    for _ in 0..100 {
        let q: Vec<f32> = (0..6).map(|_| rng.range(-3.0, 3.0)).collect();
        for link in ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        {
            worst = worst
                .max(difference(affine(usd.link_pose(&q, link).unwrap()), affine(mjcf.link_pose(&q, link).unwrap())));
        }
    }
    assert!(worst < 1e-5, "USD and MJCF UR5e differ by {worst}");
    assert!(!usd.collision_model().spheres.is_empty(), "capsule colliders give spheres");
}

#[test]
fn newton_2f85_mimics_its_driver_and_skips_loop_closures() {
    let robot = Robot::load(
        common::asset("newton/robotiq_2f85_v4/usd_structured/Dual_wrist_camera.usda"),
        &RobotOptions::default(),
    )
    .unwrap();
    // right_driver_joint follows left_driver_joint through NewtonMimicAPI; the four-bar loops,
    // closed by spherical joints, are not part of the tree.
    assert!(robot.joint_names().iter().any(|j| j == "left_driver_joint"));
    assert!(!robot.joint_names().iter().any(|j| j == "right_driver_joint"));
    let left = robot.joint_names().iter().position(|j| j == "left_driver_joint").unwrap();
    let mut q = robot.default_q().to_vec();
    let before = robot.link_pose(&q, "right_driver").unwrap();
    q[left] = 0.5;
    let after = robot.link_pose(&q, "right_driver").unwrap();
    assert!((before.rotation.angle_between(after.rotation) - 0.5).abs() < 1e-3, "the right driver does not follow");
    // Mesh colliders from the binary geometry layer give spheres.
    assert!(robot.collision_model().spheres.contains_key("base"));
}

#[test]
fn usd_scenes_load_with_schema_defaults() {
    let world = World::load(common::asset("usd/scene.usda"), &SdfOptions::default()).unwrap();
    assert_eq!(world.obstacles.len(), 4, "the visual-only sphere has no CollisionAPI");
    let probe = |p: Vec3| world.obstacles.iter().map(|o| o.distance(p).0).fold(f32::INFINITY, f32::min);
    // Floor: a default cube (size 2) scaled to 2 m x 2 cm x 2 m, its top at z = 0.
    assert!(probe(Vec3::new(0.2, 0.3, 0.0)).abs() < 1e-6);
    // Unauthored sizes in centimeters: sphere radius 1, cylinder radius 1 and height 2 (along
    // the stage's up axis), capsule radius 0.5 and height 1.
    let sphere = Vec3::new(0.5, 0.0, 0.01);
    assert!((probe(sphere) + 0.01).abs() < 1e-6, "sphere center is 1 cm inside");
    let post = Vec3::new(0.0, -0.5, 0.01);
    assert!((probe(post + Vec3::new(0.0, 0.0, 0.015)) - 0.005).abs() < 1e-6, "cylinder top is 1 cm above its center");
    let pill = Vec3::new(-0.5, 0.0, 0.01);
    assert!((probe(pill) + 0.005).abs() < 1e-6, "capsule radius is 0.5 cm");
    // The capsule's axis is the stage's z, which turns into the world's -y.
    assert!((probe(pill + Vec3::new(0.0, 0.0105, 0.0)) - 0.0005).abs() < 1e-6);
}

#[test]
fn franka_usd_converted_from_our_urdf_matches_it() {
    // assets/franka/usd was converted from assets/franka/franka_panda.urdf with NVIDIA's
    // urdf-usd-converter 0.3.3. Link names survive, so the URDF's collision model applies too.
    let urdf = common::panda().unwrap();
    let usd = Robot::load(common::asset("franka/usd/panda.usda"), &common::panda_options()).unwrap();
    assert_eq!(usd.joint_names(), urdf.joint_names());
    for (a, b) in [(usd.lower(), urdf.lower()), (usd.upper(), urdf.upper()), (usd.max_velocity(), urdf.max_velocity())]
    {
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5), "{a:?} vs {b:?}");
    }
    let links =
        ["panda_link0", "panda_link3", "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger", "ee_link"];
    let mut rng = Rng::new(10);
    let mut worst = 0.0f64;
    let mut q_all = vec![];
    for _ in 0..100 {
        let q: Vec<f32> = (0..7).map(|j| rng.range(urdf.lower()[j], urdf.upper()[j])).collect();
        for link in links {
            worst = worst
                .max(difference(affine(usd.link_pose(&q, link).unwrap()), affine(urdf.link_pose(&q, link).unwrap())));
        }
        q_all.extend(q);
    }
    assert!(worst < 1e-5, "USD and URDF Franka differ by {worst}");
    let (a, b) = (Device::cpu(&usd), Device::cpu(&urdf));
    let item_world = vec![0; 100];
    let (ea, eb) = (
        a.evaluate(&a.upload(&[World::default()]).unwrap(), &item_world, &q_all, &CollisionWeights::NONE).unwrap(),
        b.evaluate(&b.upload(&[World::default()]).unwrap(), &item_world, &q_all, &CollisionWeights::NONE).unwrap(),
    );
    assert!(ea.self_clearance.iter().zip(&eb.self_clearance).all(|(x, y)| (x - y).abs() < 1e-5));
}
