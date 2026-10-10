//! RRT-Connect, shortcutting, and the planner falling back to them. GPU parts are skipped when no
//! adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::rrt::connect;
use batchplan::shortcut::shortcut;
use batchplan::*;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot)];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// UR5e motions from the default pose (arm straight out over the table) to far goals in tabletop
/// worlds. Joints are bounded, not wrapped, so a goal whose shoulder has turned past the table
/// below cannot be reached; `above_table` keeps the shoulder within the half turn above it.
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
    let cpu = Device::cpu(&robot);
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
    let cpu = Device::cpu(&robot);
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
    let cpu = Device::cpu(&robot);
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
    let cpu = Device::cpu(&robot);
    let n = robot.dof();
    let (scene, problems) = far_reaches(&robot, &cpu, 24, false);
    let on_cpu = cpu.upload(&scene).unwrap();
    let alone = PlanOptions { fallback: None, ..Default::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let without = plan(&d, &worlds, &problems, &alone).unwrap();
        let with = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
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
            let trajectory = Trajectory::new(&robot, s.solution, 1.0);
            trajectory.check(&robot).unwrap();
            let samples = trajectory.sample(32.0 / trajectory.knot_interval).positions;
            item_world.extend(std::iter::repeat_n(s.problem.world, samples.len() / n));
            q.extend(samples);
        }
        let e = cpu.evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE).unwrap();
        let worst = e.world_clearance.iter().chain(&e.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v));
        assert!(worst > -2e-3, "{}: a plan penetrates by {worst}", d.name());
    }
}
