//! RRT-Connect, shortcutting, and the planner falling back to them. GPU parts are skipped when no
//! adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::rrt::connect;
use batchplan::shortcut::shortcut;
use batchplan::*;
use glam::Vec3;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// UR5e motions from the default pose (arm straight out over the table) to far goals in tabletop
/// worlds; `above_table` keeps the shoulder within the half turn above the table.
fn far_reaches(robot: &Robot, cpu: &Device, count: usize, above_table: bool) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = Rng::new(14);
    let scene: Vec<World> = (0..4).map(|_| common::tabletop_for(robot, cpu, &mut rng)).collect();
    let worlds = cpu.upload(&scene).unwrap();
    let mut problems = vec![];
    while problems.len() < count {
        let world = problems.len() as u32 % 4;
        let mut goal: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        if above_table {
            goal[1] = rng.range(-3.0, -0.1);
        }
        let far = goal.iter().zip(robot.default_q()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt() > 4.0;
        if far && cpu.evaluate(&worlds, &[world], &goal, &CollisionWeights::NONE).unwrap().collision_free(0) {
            problems.push(PlanProblem { world, start: robot.default_q().to_vec(), goal });
        }
    }
    (scene, problems)
}

fn searches(problems: &[PlanProblem]) -> Vec<RrtProblem> {
    problems
        .iter()
        .map(|p| RrtProblem { world: p.world, start: p.start.clone(), goals: vec![p.goal.clone()] })
        .collect()
}

fn length(path: &[f32], n: usize) -> f32 {
    path.chunks(n)
        .zip(path.chunks(n).skip(1))
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt())
        .sum()
}

/// Every edge of every path sampled at `resolution` on the CPU: the worst clearance.
fn worst_clearance(cpu: &Device, worlds: &Worlds, paths: &[(u32, &[f32])], resolution: f32) -> f32 {
    let n = cpu.robot().dof();
    let (mut q, mut item_world) = (vec![], vec![]);
    for &(world, path) in paths {
        for (a, b) in path.chunks(n).zip(path.chunks(n).skip(1)) {
            let d = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
            let count = (d / resolution).ceil().max(1.0) as usize;
            for k in 0..=count {
                let t = k as f32 / count as f32;
                q.extend(a.iter().zip(b).map(|(x, y)| x + (y - x) * t));
                item_world.push(world);
            }
        }
    }
    let e = cpu.evaluate(worlds, &item_world, &q, &CollisionWeights::NONE).unwrap();
    e.world_clearance.iter().chain(&e.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v))
}

#[test]
fn rrt_paths_are_collision_free_under_a_denser_check() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = far_reaches(&robot, &cpu, 12, true);
    let on_cpu = cpu.upload(&scene).unwrap();
    let o = RrtOptions::default();
    for d in devices(&robot) {
        let found = connect(&d, &d.upload(&scene).unwrap(), &searches(&problems), &o).unwrap();
        let mut paths = vec![];
        for (p, path) in problems.iter().zip(&found.paths) {
            let Some(path) = path else { continue };
            assert!(path[..n] == p.start[..] && path[path.len() - n..] == p.goal[..], "{}: wrong ends", d.name());
            let longest = path
                .chunks(n)
                .zip(path.chunks(n).skip(1))
                .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt())
                .fold(0.0f32, f32::max);
            assert!(longest <= o.step * 1.0001, "{}: an edge of {longest}", d.name());
            paths.push((p.world, &path[..]));
        }
        eprintln!("{}: RRT-Connect found {} of {} paths", d.name(), paths.len(), problems.len());
        assert!(paths.len() >= 10, "{}: found only {} paths", d.name(), paths.len());
        let worst = worst_clearance(&cpu, &on_cpu, &paths, o.resolution / 4.0);
        assert!(worst > -1e-3, "{}: an edge penetrates by {worst}", d.name());
    }
}

#[test]
fn shortcutting_never_lengthens_a_path() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = far_reaches(&robot, &cpu, 12, true);
    let worlds = cpu.upload(&scene).unwrap();
    let found = connect(&cpu, &worlds, &searches(&problems), &RrtOptions::default()).unwrap();
    let (world, raw): (Vec<u32>, Vec<Vec<f32>>) =
        problems.iter().zip(found.paths).filter_map(|(p, path)| Some((p.world, path?))).unzip();
    let mut short = raw.clone();
    let o = ShortcutOptions::default();
    shortcut(&cpu, &worlds, &world, &mut short, &o).unwrap();
    let (mut before, mut after) = (0.0, 0.0);
    for (a, b) in raw.iter().zip(&short) {
        assert!(length(b, n) <= length(a, n), "shortcutting lengthened a path: {} > {}", length(b, n), length(a, n));
        assert!(a[..n] == b[..n] && a[a.len() - n..] == b[b.len() - n..], "shortcutting moved an end");
        (before, after) = (before + length(a, n), after + length(b, n));
    }
    let straight: f32 = raw.iter().map(|p| length(&[&p[..n], &p[p.len() - n..]].concat(), n)).sum();
    eprintln!("total length {before:.1} -> {after:.1} over {} paths; straight lines {straight:.1}", raw.len());
    // Straight lines from start to goal bound the length from below.
    assert!(after < before && after < 1.1 * straight, "shortcutting barely helped: {before} -> {after}");
    let paths: Vec<(u32, &[f32])> = world.iter().copied().zip(short.iter().map(|p| &p[..])).collect();
    let worst = worst_clearance(&cpu, &worlds, &paths, o.resolution / 4.0);
    assert!(worst > -1e-3, "a shortcut penetrates by {worst}");
}

#[test]
fn fixed_seeds_give_fixed_results() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let (scene, problems) = far_reaches(&robot, &cpu, 6, true);
    let worlds = cpu.upload(&scene).unwrap();
    let run = |rrt: RrtOptions, short: ShortcutOptions| {
        let found = connect(&cpu, &worlds, &searches(&problems), &rrt).unwrap();
        let (world, mut paths): (Vec<u32>, Vec<Vec<f32>>) =
            problems.iter().zip(found.paths).filter_map(|(p, path)| Some((p.world, path?))).unzip();
        shortcut(&cpu, &worlds, &world, &mut paths, &short).unwrap();
        paths
    };
    let (rrt, short) = (RrtOptions::default(), ShortcutOptions::default());
    let first = run(rrt, short);
    assert!(!first.is_empty());
    assert_eq!(first, run(rrt, short));
    assert_ne!(first, run(RrtOptions { rng_seed: rrt.rng_seed + 1, ..rrt }, short), "the seed changes nothing");
    assert_ne!(first, run(rrt, ShortcutOptions { rng_seed: short.rng_seed + 1, ..short }));
}

#[test]
fn plan_falls_back_to_rrt_where_trajectory_optimization_fails() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = far_reaches(&robot, &cpu, 24, false);
    let on_cpu = cpu.upload(&scene).unwrap();
    // Trajectory optimization cut short, so the fallback has problems to solve.
    let weak = PlanOptions { seeds: 2, iterations: 8, ..Default::default() };
    let alone = PlanOptions { fallback: None, ..weak };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let without = plan(&d, &worlds, &problems, &alone).unwrap();
        let with = plan(&d, &worlds, &problems, &weak).unwrap();
        let reachable = connect(&d, &worlds, &searches(&problems), &RrtOptions::default()).unwrap();
        let (a, b) = (without.solved().count(), with.solved().count());
        let rrt = reachable.paths.iter().flatten().count();
        eprintln!("{}: trajectory optimization alone {a}/24, with the fallback {b}/24, RRT-Connect {rrt}/24", d.name());
        assert!(b > a, "{}: the fallback solved nothing more ({a})", d.name());
        for (i, path) in reachable.paths.iter().enumerate() {
            assert!(
                path.is_none() || with.best(i).is_some(),
                "{}: RRT-Connect reaches problem {i}; the plan does not",
                d.name()
            );
        }
        // Problems trajectory optimization solved keep their paths.
        for s in without.solved() {
            assert_eq!(with.best(s.index), Some(s.solution));
        }
        // Every fallback path, timed and sampled 4x denser than validation, under the CPU model.
        let (mut q, mut item_world) = (vec![], vec![]);
        for s in with.solved() {
            let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
            trajectory.check(&robot).unwrap();
            let samples = trajectory.sample(32.0 / trajectory.knot_interval).unwrap().positions;
            item_world.extend(std::iter::repeat_n(s.problem.world, samples.len() / n));
            q.extend(samples);
        }
        let e = cpu.evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE).unwrap();
        let worst = e.world_clearance.iter().chain(&e.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v));
        assert!(worst > -2e-3, "{}: a plan penetrates by {worst}", d.name());
    }
}

#[test]
fn the_fallback_keeps_the_time_budget() {
    // Trajectory optimization cut short leaves most problems to RRT-Connect, whose fine steps make
    // the search long enough to cut.
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let (scene, problems) = far_reaches(&robot, &cpu, 24, false);
    let rrt = RrtOptions { step: 0.02, ..Default::default() };
    let slow = PlanOptions {
        seeds: 1,
        iterations: 2,
        fallback: Some(Fallback { rrt, ..Default::default() }),
        ..Default::default()
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let timed = |o: &PlanOptions| {
            let t = std::time::Instant::now();
            plan(&d, &worlds, &problems, o).unwrap();
            t.elapsed()
        };
        let full = timed(&slow);
        let alone = timed(&PlanOptions { fallback: None, ..slow });
        // The budget runs out during the search, after trajectory optimization.
        let budget = alone + (full - alone) / 4;
        let spent = timed(&PlanOptions { time_budget: Some(budget), ..slow });
        eprintln!("{}: full {full:?}, without fallback {alone:?}, budget {budget:?} -> {spent:?}", d.name());
        let search = full - alone;
        assert!(
            search > std::time::Duration::from_millis(100),
            "{}: the search is too short to cut ({search:?})",
            d.name()
        );
        assert!(spent < budget + search / 4, "{}: took {spent:?} of a {budget:?} budget", d.name());
    }
}

/// A turntable arm: a continuous joint about z carrying a 40 cm arm along x, and a shoulder.
const TURNTABLE: &str = r#"<robot name="turntable">
  <link name="base"/>
  <link name="table"><collision><origin xyz="0.2 0 0.05"/><geometry><box size="0.4 0.06 0.06"/></geometry></collision></link>
  <link name="arm"><collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.05 0.05"/></geometry></collision></link>
  <joint name="turn" type="continuous"><parent link="base"/><child link="table"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="table"/><child link="arm"/><origin xyz="0.4 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
</robot>"#;

/// How far each joint moves along `path` (`[k, dof]` waypoints or control points).
fn travel(path: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|j| path.chunks(n).zip(path.chunks(n).skip(1)).map(|(a, b)| (b[j] - a[j]).abs()).sum()).collect()
}

#[test]
fn continuous_joints_turn_the_short_way_unless_it_is_blocked() {
    let dir = std::env::temp_dir().join(format!("batchplan-turntable-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("turntable.urdf"), TURNTABLE).unwrap();
    let robot = Robot::load(dir.join("turntable.urdf"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.continuous(), [true, false]);
    // From 2.6 rad to -2.6 rad is 1.08 rad through pi, or 5.2 rad the long way.
    let problem = PlanProblem { world: 0, start: vec![2.6, 0.0], goal: vec![-2.6, 0.0] };
    let short = std::f32::consts::TAU - 5.2;
    // A fin along -x, beyond the turntable but within the arm's reach: across the short way only
    // (the arm points along -x at pi).
    let wall = World {
        obstacles: vec![Obstacle::Cuboid {
            center: Vec3::new(-0.55, 0.0, 0.15),
            half_extents: Vec3::new(0.1, 0.01, 0.25),
            rotation: glam::Quat::IDENTITY,
        }],
    };
    for d in devices(&robot) {
        let worlds = d.upload(&[World::default(), wall.clone()]).unwrap();
        let open = plan(&d, &worlds, std::slice::from_ref(&problem), &PlanOptions::default()).unwrap();
        let path = open.best(0).unwrap_or_else(|| panic!("{}: no plan", d.name()));
        let turned = travel(path, 2)[0];
        assert!((turned - short).abs() < 0.05, "{}: turned {turned} rad, not the short way {short}", d.name());
        // The trajectory runs past pi, as a continuous joint may.
        Trajectory::new(&robot, path, 1.0).unwrap().check(&robot).unwrap();
        // The final pose is the goal's.
        let (end, goal) = (robot.ee_pose(&path[path.len() - 2..]), robot.ee_pose(&problem.goal));
        assert!((end.position - goal.position).length() < 1e-4, "{}: ends at {end:?}", d.name());

        let blocked = PlanProblem { world: 1, ..problem.clone() };
        let around = plan(&d, &worlds, &[blocked], &PlanOptions::default()).unwrap();
        let path = around.best(0).unwrap_or_else(|| panic!("{}: no plan around the wall", d.name()));
        let turned = travel(path, 2)[0];
        assert!(turned > 4.0, "{}: turned {turned} rad; the short way is blocked", d.name());

        // With the fin along +x instead, only the way through pi is open, and RRT-Connect finds
        // it: its distances and steps wrap.
        let fin = |x: f32| World {
            obstacles: vec![Obstacle::Cuboid {
                center: Vec3::new(x, 0.0, 0.15),
                half_extents: Vec3::new(0.1, 0.01, 0.25),
                rotation: glam::Quat::IDENTITY,
            }],
        };
        let worlds = d.upload(&[fin(0.55)]).unwrap();
        let search = RrtProblem { world: 0, start: problem.start.clone(), goals: vec![problem.goal.clone()] };
        let found = connect(&d, &worlds, &[search], &RrtOptions::default()).unwrap();
        let path =
            found.paths[0].as_ref().unwrap_or_else(|| panic!("{}: RRT-Connect found no way through pi", d.name()));
        let turned = travel(path, 2)[0];
        assert!(turned < 2.0, "{}: RRT-Connect turned {turned} rad", d.name());
    }
}

#[test]
fn ur5e_reaches_a_goal_a_turn_away_the_other_way_round() {
    // The goal's shoulder pan is 4 rad, past a wall at +90 degrees; the same pose at 4 - 2 pi lies
    // the other way round. Within the joint's interval the goal cannot be reached.
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let mut start = robot.default_q().to_vec();
    start[0] = 0.0;
    let mut goal = start.clone();
    goal[0] = 4.0;
    let wall = World {
        obstacles: vec![Obstacle::Cuboid {
            center: Vec3::new(0.0, 0.6, 0.3),
            half_extents: Vec3::new(0.05, 0.3, 0.6),
            rotation: glam::Quat::IDENTITY,
        }],
    };
    for d in devices(&robot) {
        let worlds = d.upload(std::slice::from_ref(&wall)).unwrap();
        let problem = PlanProblem { world: 0, start: start.clone(), goal: goal.clone() };
        let result = plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
        let path = result.best(0).unwrap_or_else(|| panic!("{}: no plan", d.name()));
        let n = robot.dof();
        let end = &path[path.len() - n..];
        assert!((end[0] - (4.0 - std::f32::consts::TAU)).abs() < 1e-5, "{}: ends with pan {}", d.name(), end[0]);
        let (reached, wanted) = (robot.ee_pose(end), robot.ee_pose(&goal));
        assert!((reached.position - wanted.position).length() < 1e-4, "{}: ends at {reached:?}", d.name());
        assert!(path.chunks(n).all(|q| q[0] <= 0.0), "{}: the pan turns toward the wall", d.name());
    }
}
