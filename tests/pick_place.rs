//! Pick-and-place demonstrations: the gripper closes once and opens again, the object goes where
//! it was asked to and follows the gripper while held, and nothing collides on the way.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::datagen::{PickPlace, PickPlaceOptions, pick_and_place};
use batchplan::rng::Rng;
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

/// A table with its top at z = 0, a tall box beside the path, and a 4 x 6 x 6 cm box to move;
/// the box last, its target across the table.
fn scenes(count: usize) -> (Vec<World>, Vec<PickPlace>) {
    let mut rng = Rng::new(5);
    let (mut worlds, mut tasks) = (vec![], vec![]);
    for w in 0..count {
        let table = Obstacle::Cuboid {
            center: Vec3::new(0.4, 0.0, -0.02),
            half_extents: Vec3::new(0.7, 0.8, 0.02),
            rotation: Quat::IDENTITY,
        };
        let clutter = Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.55, 0.65), rng.range(-0.05, 0.05), 0.15),
            half_extents: Vec3::new(0.04, 0.04, 0.15),
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        };
        let half = Vec3::new(0.02, 0.03, 0.03);
        let object = Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.35, 0.5), rng.range(0.15, 0.3), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(-1.0, 1.0)),
        };
        let place = Pose {
            position: Vec3::new(rng.range(0.35, 0.5), rng.range(-0.3, -0.15), half.z),
            rotation: Quat::from_rotation_z(rng.range(-1.0, 1.0)),
        };
        worlds.push(World { obstacles: vec![table, clutter, object] });
        tasks.push(PickPlace { world: w as u32, object: 2, place });
    }
    (worlds, tasks)
}

#[test]
fn boxes_are_picked_up_carried_and_set_down_where_asked() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (worlds, tasks) = scenes(6);
    for d in devices(&robot) {
        let demos = pick_and_place(&d, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
        assert!(demos.len() >= 4, "{}: {} of {} tasks done", d.name(), demos.len(), tasks.len());
        eprintln!("{}: {} of {} tasks done", d.name(), demos.len(), tasks.len());
        let cpu = Device::cpu(&robot).unwrap();
        for demo in &demos {
            let task = tasks.iter().find(|t| t.world == demo.world).unwrap();
            let world = &worlds[demo.world as usize];
            let frames = demo.trajectory.len();
            let q = |f: usize| &demo.trajectory.positions[f * n..(f + 1) * n];
            // The gripper starts and ends open and closes once.
            let g = &demo.gripper;
            assert!(g[0] == 1.0 && g[frames - 1] == 1.0 && g.contains(&0.0), "{}: gripper {g:?}", d.name());
            let closings = g.windows(2).filter(|w| w[0] > 0.0 && w[1] == 0.0).count();
            assert_eq!(closings, 1, "{}: the gripper closes once", d.name());
            // It moves over the default half second, never in a jump.
            let step = demo.trajectory.dt / 0.5 + 1e-5;
            assert!(g.windows(2).all(|w| (w[1] - w[0]).abs() <= step), "{}: the gripper jumps", d.name());
            // The object starts where it is and ends where it was asked to go.
            let carried = demo.carried.as_ref().unwrap();
            assert_eq!(carried.obstacle, task.object);
            assert_eq!(carried.poses[0], world.obstacles[task.object].pose());
            let end = carried.poses[frames - 1];
            assert!((end.position - task.place.position).length() < 3e-3, "{}: set down at {}", d.name(), end.position);
            assert!(end.rotation.dot(task.place.rotation).abs() > 1.0 - 1e-4, "{}: set down turned", d.name());
            // While held, it keeps its place in the gripper.
            let held: Vec<usize> = (0..frames).filter(|&f| g[f] == 0.0).collect();
            let grip = |f: usize| robot.ee_pose(q(f)).inverse().mul_pose(carried.poses[f]);
            let first = grip(held[0]);
            for &f in &held {
                let now = grip(f);
                assert!((now.position - first.position).length() < 1e-4, "{}: frame {f} slips", d.name());
            }
            // The arm never touches the table or the clutter, and the box, once lifted 2 cm,
            // touches nothing either.
            let mut without = world.clone();
            without.obstacles.remove(task.object);
            let uploaded = cpu.upload(std::slice::from_ref(&without)).unwrap();
            let clear =
                cpu.evaluate(&uploaded, &vec![0; frames], &demo.trajectory.positions, &CollisionWeights::NONE).unwrap();
            assert!((0..frames).all(|f| clear.collision_free(f)), "{}: the arm collides", d.name());
            let mut lifted = without.clone();
            let start_z = carried.poses[0].position.z;
            let up: Vec<usize> =
                held.iter().copied().filter(|&f| carried.poses[f].position.z > start_z + 0.02).collect();
            assert!(!up.is_empty(), "{}: the box should be lifted", d.name());
            for &f in &up {
                lifted.obstacles = without.obstacles.clone();
                lifted.obstacles.push(world.obstacles[task.object].placed(carried.poses[f]));
                let box_now = lifted.obstacles.last().unwrap();
                let nearest = without
                    .obstacles
                    .iter()
                    .map(|o| {
                        // The box's corners against each obstacle: a cheap check that it clears them.
                        (0..8)
                            .map(|c| {
                                let Obstacle::Cuboid { center, half_extents: h, rotation } = *box_now else {
                                    unreachable!()
                                };
                                let corner =
                                    Vec3::new([-h.x, h.x][c & 1], [-h.y, h.y][c >> 1 & 1], [-h.z, h.z][c >> 2 & 1]);
                                o.distance(center + rotation * corner).0
                            })
                            .fold(f32::INFINITY, f32::min)
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(nearest > 0.0, "{}: frame {f}: the carried box touches the scene ({nearest})", d.name());
            }
            // The grasp point is inside the box, 2 cm below its top.
            let Obstacle::Cuboid { half_extents: half, .. } = world.obstacles[task.object] else { unreachable!() };
            let tool = first.inverse().position;
            assert!(
                tool.x.abs() < 1e-3 && tool.y.abs() < 1e-3 && (tool.z - (half.z - 0.02)).abs() < 1e-3,
                "{}: held at {tool}",
                d.name()
            );
            // The fingers close across the box's narrower side, its x axis.
            let closing = robot.ee_pose(q(held[0])).rotation * Vec3::Y;
            let narrow = world.obstacles[task.object].pose().rotation * Vec3::X;
            assert!(closing.dot(narrow).abs() > 0.999, "{}: closing along {closing}", d.name());
            // Velocities point the way the positions go, played backwards or not.
            let v = |f: usize| &demo.trajectory.velocities[f * n..(f + 1) * n];
            for f in 1..frames - 1 {
                for j in 0..n {
                    let moved = (q(f + 1)[j] - q(f - 1)[j]) / (2.0 * demo.trajectory.dt);
                    assert!(
                        moved.abs() < 0.05 || moved.signum() == v(f)[j].signum(),
                        "{}: frame {f} joint {j} moves {moved} at {}",
                        d.name(),
                        v(f)[j]
                    );
                }
            }
            // Speeds stay within the robot's limits.
            for f in 1..frames {
                for j in 0..n {
                    let speed = (q(f)[j] - q(f - 1)[j]).abs() / demo.trajectory.dt;
                    assert!(speed <= robot.max_velocity()[j] * 1.05, "{}: joint {j} at {speed} rad/s", d.name());
                }
            }
            assert!(
                demo.task.starts_with("Pick up the box and put it") && demo.task.ends_with("to the right."),
                "{}",
                demo.task
            );
        }
    }
}

#[test]
fn pick_and_place_tasks_are_checked() {
    let robot = common::panda().unwrap();
    let d = Device::cpu(&robot).unwrap();
    let (worlds, tasks) = scenes(1);
    let wrong = |t: PickPlace| pick_and_place(&d, &worlds, &[t], &PickPlaceOptions::default()).is_err();
    assert!(wrong(PickPlace { world: 1, ..tasks[0].clone() }), "a world past the end");
    assert!(wrong(PickPlace { object: 0, ..tasks[0].clone() }), "the table is too wide to grasp");
    let tilted = Pose { rotation: Quat::from_rotation_x(0.5), ..tasks[0].place };
    assert!(wrong(PickPlace { place: tilted, ..tasks[0].clone() }), "a tilted place pose");
}
