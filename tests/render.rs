//! Camera images: depths against closed forms for every obstacle kind and the robot, shading,
//! wrist cameras, moved obstacles, and GPU against CPU.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use batchplan::*;
use glam::{Quat, Vec3};

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// 64 x 48 pixels, the principal point on pixel (32, 24).
fn camera(mount: Mount) -> Camera {
    Camera {
        name: "test".into(),
        width: 64,
        height: 48,
        intrinsics: Intrinsics { fx: 60.0, fy: 60.0, cx: 32.0, cy: 24.0 },
        mount,
    }
}

/// 1 m above (0.6, 0, 0), looking straight down, beside the robot.
fn overhead() -> Mount {
    Mount::World(Pose { position: Vec3::new(0.6, 0.0, 1.0), rotation: Quat::from_rotation_x(PI) })
}

const CENTRE: usize = 24 * 64 + 32;

/// A configuration that turns the Panda's arm away from the overhead camera.
fn aside(robot: &Robot) -> Vec<f32> {
    let mut q = robot.default_q().to_vec();
    q[0] = PI * 0.9;
    q
}

#[test]
fn every_obstacle_kind_is_seen_at_its_depth() {
    let robot = common::panda().unwrap();
    let (vertices, triangles) = cube(0.1);
    let grid = Arc::new(SdfGrid::from_mesh(&vertices, &triangles, &SdfOptions::default()).unwrap());
    let lying = Quat::from_rotation_y(FRAC_PI_2);
    // Each world holds one obstacle under the camera and the depth of its top.
    let cases = [
        (Obstacle::Sphere { center: Vec3::new(0.6, 0.0, 0.3), radius: 0.1 }, 0.6),
        (
            Obstacle::Cuboid {
                center: Vec3::new(0.6, 0.0, 0.2),
                half_extents: Vec3::new(0.1, 0.2, 0.15),
                rotation: Quat::from_rotation_z(0.3),
            },
            0.65,
        ),
        (
            Obstacle::Cylinder {
                center: Vec3::new(0.6, 0.0, 0.2),
                rotation: Quat::IDENTITY,
                radius: 0.05,
                half_height: 0.25,
            },
            0.55,
        ),
        (
            Obstacle::Cylinder { center: Vec3::new(0.6, 0.0, 0.2), rotation: lying, radius: 0.05, half_height: 0.25 },
            0.75,
        ),
        (
            Obstacle::Capsule {
                center: Vec3::new(0.6, 0.0, 0.2),
                rotation: Quat::IDENTITY,
                radius: 0.05,
                half_length: 0.1,
            },
            0.65,
        ),
        (Obstacle::Capsule { center: Vec3::new(0.6, 0.0, 0.2), rotation: lying, radius: 0.05, half_length: 0.1 }, 0.75),
        (Obstacle::Sdf { grid, center: Vec3::new(0.6, 0.0, 0.3), rotation: Quat::IDENTITY }, 0.6),
    ];
    let scene: Vec<World> = cases.iter().map(|(o, _)| World { obstacles: vec![o.clone()] }).collect();
    let q = aside(&robot);
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let items: Vec<u32> = (0..cases.len() as u32).collect();
        let configurations = q.repeat(cases.len());
        let images = d.render(&worlds, &camera(overhead()), &items, &configurations, &[]).unwrap();
        // Each top faces straight up, so it takes the first obstacle colour lit from above.
        let light = Vec3::new(0.3, 0.2, 1.0).normalize();
        let shade = [0.55, 0.42, 0.30].map(|c: f32| (c * (0.35 + 0.65 * light.z) * 255.0).round() as u8);
        for (i, (obstacle, depth)) in cases.iter().enumerate() {
            let seen = images.depth[i * 64 * 48 + CENTRE];
            // The grid's surface sits up to its offset (under a voxel diagonal) outside the cube,
            // and its normal is an interpolated gradient.
            let grid = matches!(obstacle, Obstacle::Sdf { .. });
            let tolerance = if grid { 0.02 } else { 1e-4 };
            assert!(
                seen <= depth + 1e-4 && seen > depth - tolerance,
                "{}: {obstacle:?} at {seen}, not {depth}",
                d.name()
            );
            let rgb = &images.rgb[(i * 64 * 48 + CENTRE) * 3..][..3];
            let off = rgb.iter().zip(&shade).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= if grid { 3 } else { 1 }, "{}: {obstacle:?} coloured {rgb:?}, not {shade:?}", d.name());
        }
        // Past the sphere's edge the background shows.
        // At depth 0.6 the sphere spans 0.1 / 0.6 * 60 = 10 pixels from the centre (tangent rays
        // shrink that a little).
        let row = |u: usize| images.depth[24 * 64 + u];
        assert!(row(32 + 8) > 0.0 && row(32 + 11) == 0.0, "{}: the sphere's edge", d.name());
    }
}

/// A closed cube mesh of side `2 * half`, centred at the origin.
fn cube(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let vertices =
        (0..8).map(|i| Vec3::new([-half, half][i & 1], [-half, half][i >> 1 & 1], [-half, half][i >> 2 & 1])).collect();
    let faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let triangles = faces.iter().flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]]).collect();
    (vertices, triangles)
}

#[test]
fn the_robot_is_drawn_from_its_spheres() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let eye = Pose {
        position: Vec3::new(1.2, 0.0, 0.6),
        rotation: look_at(Vec3::new(1.2, 0.0, 0.6), Vec3::new(0.2, 0.0, 0.4)),
    };
    let cam = camera(Mount::World(eye));
    // The robot's depth from an independent ray caster.
    let blank = vec![0.0f32; 64 * 48];
    let image = DepthImage { depth: &blank, width: 64, intrinsics: cam.intrinsics, camera: eye };
    let expected = image.robot_depth(&robot, &q, 0.0).unwrap();
    let robot_pixels = expected.iter().filter(|d| d.is_finite()).count();
    assert!(robot_pixels > 200, "the robot should fill part of the view: {robot_pixels} pixels");
    for d in devices(&robot) {
        let worlds = d.upload(&[World::default()]).unwrap();
        let images = d.render(&worlds, &cam, &[0], &q, &[]).unwrap();
        // The two casters work in different frames; near a sphere's edge the root is sensitive.
        for (p, &e) in expected.iter().enumerate() {
            let seen = images.depth[p];
            if e.is_finite() {
                assert!((seen - e).abs() < 5e-4, "{}: pixel {p} at {seen}, not {e}", d.name());
                let [r, g, b] = [0, 1, 2].map(|c| images.rgb[p * 3 + c]);
                assert!(r == g && r > 50, "{}: pixel {p} should be the robot's grey, is {r} {g} {b}", d.name());
            } else {
                assert_eq!(seen, 0.0, "{}: pixel {p}", d.name());
            }
        }
    }
}

/// The rotation of a camera at `eye` looking at `target` (OpenCV axes).
fn look_at(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    Quat::from_mat3(&glam::Mat3::from_cols(right, forward.cross(right), forward))
}

#[test]
fn wrist_cameras_follow_the_arm_and_moved_obstacles_move() {
    let robot = common::panda().unwrap();
    // 15 cm along the hand's z axis, past the fingertips, looking along it.
    let offset = Pose { position: Vec3::new(0.0, 0.0, 0.15), rotation: Quat::IDENTITY };
    let wrist = camera(Mount::Link { link: "panda_hand".into(), offset });
    let poses: Vec<Vec<f32>> = [0.0, 0.6]
        .iter()
        .map(|&t| robot.default_q().iter().enumerate().map(|(j, &v)| v + if j == 0 { t } else { 0.0 }).collect())
        .collect();
    // A ball 0.5 m in front of each pose's camera, in its own world.
    let scene: Vec<World> = poses
        .iter()
        .map(|q| {
            let eye = robot.link_pose(q, "panda_hand").unwrap().mul_pose(offset);
            World {
                obstacles: vec![Obstacle::Sphere { center: eye.position + eye.rotation * Vec3::Z * 0.5, radius: 0.05 }],
            }
        })
        .collect();
    let q: Vec<f32> = poses.concat();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let images = d.render(&worlds, &wrist, &[0, 1], &q, &[]).unwrap();
        for v in 0..2 {
            let seen = images.depth[v * 64 * 48 + CENTRE];
            assert!((seen - 0.45).abs() < 1e-4, "{}: view {v} sees the ball at {seen}", d.name());
        }
        // Moving the ball 10 cm farther along the camera's axis moves its depth with it.
        let eye = robot.link_pose(&poses[0], "panda_hand").unwrap().mul_pose(offset);
        let farther = Pose { position: eye.position + eye.rotation * Vec3::Z * 0.6, rotation: Quat::IDENTITY };
        let images = d.render(&worlds, &wrist, &[0], &poses[0], &[Some((0, farther))]).unwrap();
        assert!((images.depth[CENTRE] - 0.55).abs() < 1e-4, "{}: the moved ball at {}", d.name(), images.depth[CENTRE]);
        // Malformed requests are errors.
        assert!(d.render(&worlds, &wrist, &[0], &poses[0], &[Some((1, farther))]).is_err(), "an obstacle past the end");
        let nowhere = camera(Mount::Link { link: "nowhere".into(), offset });
        assert!(d.render(&worlds, &nowhere, &[0], &poses[0], &[]).is_err(), "an unknown link");
        assert!(
            d.render(&worlds, &Camera { width: 0, ..wrist.clone() }, &[0], &poses[0], &[]).is_err(),
            "an empty image"
        );
    }
}

#[test]
fn the_gpu_draws_what_the_cpu_draws() {
    let robot = common::panda().unwrap();
    let devices = devices(&robot);
    if devices.len() < 2 {
        return;
    }
    let mut rng = batchplan::rng::Rng::new(4);
    let (vertices, triangles) = cube(0.06);
    let grid = Arc::new(SdfGrid::from_mesh(&vertices, &triangles, &SdfOptions::default()).unwrap());
    let mut scene: Vec<World> = (0..4).map(|_| common::tabletop(&mut rng)).collect();
    for w in &mut scene {
        w.obstacles.push(Obstacle::Capsule {
            center: Vec3::new(0.5, -0.3, 0.3),
            rotation: Quat::from_rotation_x(1.0),
            radius: 0.04,
            half_length: 0.1,
        });
        w.obstacles.push(Obstacle::Cylinder {
            center: Vec3::new(0.3, 0.4, 0.2),
            rotation: Quat::from_rotation_y(0.7),
            radius: 0.05,
            half_height: 0.12,
        });
        w.obstacles.push(Obstacle::Sdf {
            grid: grid.clone(),
            center: Vec3::new(0.6, 0.0, 0.5),
            rotation: Quat::from_rotation_z(0.4),
        });
    }
    let eye = Vec3::new(1.4, 0.6, 1.0);
    let cam = Camera {
        width: 160,
        height: 120,
        intrinsics: Intrinsics { fx: 140.0, fy: 140.0, cx: 79.5, cy: 59.5 },
        ..camera(Mount::World(Pose { position: eye, rotation: look_at(eye, Vec3::new(0.4, 0.0, 0.2)) }))
    };
    let q: Vec<f32> = (0..4).flat_map(|_| robot.default_q().to_vec()).collect();
    let moved = [
        None,
        Some((1, Pose { position: Vec3::new(0.5, 0.2, 0.4), rotation: Quat::from_rotation_z(0.5) })),
        None,
        None,
    ];
    let renders: Vec<Images> = devices
        .iter()
        .map(|d| d.render(&d.upload(&scene).unwrap(), &cam, &[0, 1, 2, 3], &q, &moved).unwrap())
        .collect();
    let (a, b) = (&renders[0], &renders[1]);
    let pixels = a.depth.len();
    let hit = a.depth.iter().filter(|&&d| d > 0.0).count();
    assert!(hit > pixels / 2, "the scene should fill the view: {hit} of {pixels}");
    // Rays that graze an edge may land either way in floating point.
    let differ = (0..pixels).filter(|&p| (a.depth[p] - b.depth[p]).abs() > 1e-3).count();
    assert!(differ * 200 < pixels, "{differ} of {pixels} depths differ");
    let colors = (0..pixels).filter(|&p| (0..3).any(|c| a.rgb[p * 3 + c].abs_diff(b.rgb[p * 3 + c]) > 2)).count();
    assert!(colors * 200 < pixels, "{colors} of {pixels} colours differ");
}
