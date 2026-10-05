//! Behaviour every `Device` must share, checked on each available device (the GPU is skipped when
//! absent unless `BATCHPLAN_REQUIRE_GPU=1`).

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::*;
use glam::Vec3;

fn panda() -> Robot {
    Robot::from_config_file(common::panda_config()).unwrap()
}

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot)];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

#[test]
fn malformed_batches_are_errors_on_every_device() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let worlds = vec![World { obstacles: vec![Obstacle::Sphere { center: Vec3::new(2.0, 0.0, 0.0), radius: 0.1 }] }];
    let none = CollisionWeights::NONE;
    for d in devices(&robot) {
        let name = d.name();
        assert!(d.evaluate(&worlds, &[0, 0], &q, &none).is_err(), "{name}: two items, one configuration");
        assert!(d.evaluate(&worlds, &[0], &q[..6], &none).is_err(), "{name}: configuration missing a joint");
        assert!(d.evaluate(&worlds, &[1], &q, &none).is_err(), "{name}: world index past the end");
        let target = robot.ee_pose(&q);
        assert!(solve_ik(&d, &worlds, &[IkProblem { world: 1, target }], &IkOptions::default()).is_err(), "{name}: IK");
        let problem = PlanProblem { world: 1, start: q.clone(), goal: q.clone() };
        assert!(plan(&d, &worlds, &[problem], &PlanOptions::default()).is_err(), "{name}: plan");
    }
}

#[test]
fn obstacle_free_worlds_work_on_every_device() {
    let robot = panda();
    let worlds = vec![World::default()];
    let start = robot.default_q().to_vec();
    let mut elsewhere = start.clone();
    elsewhere[0] += 0.8;
    elsewhere[3] += 0.4;
    let target = robot.ee_pose(&elsewhere);
    for d in devices(&robot) {
        let name = d.name();
        let e = d.evaluate(&worlds, &[0], &start, &CollisionWeights::NONE).unwrap();
        assert!(e.world_clearance[0] > 1e29, "{name}: an empty world has nothing to hit");
        let ik = solve_ik(&d, &worlds, &[IkProblem { world: 0, target }], &IkOptions::default()).unwrap();
        let goal = ik.best(0).unwrap_or_else(|| panic!("{name}: reachable target unsolved")).to_vec();
        let result =
            plan(&d, &worlds, &[PlanProblem { world: 0, start: start.clone(), goal }], &PlanOptions::default())
                .unwrap();
        assert!(result.best(0).is_some(), "{name}: no valid plan in an empty world");
    }
}

#[test]
fn results_keep_the_worlds_of_their_problems() {
    // Two targets in world 1 and one in world 0: problem index != world index.
    let robot = panda();
    let mut rng = batchplan::rng::Rng::new(21);
    let worlds: Vec<World> = (0..2).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = [1u32, 1, 0]
        .iter()
        .map(|&w| IkProblem { world: w, target: common::grasp_target(&worlds[w as usize], &mut rng) })
        .collect();
    let cpu = Device::cpu(&robot);
    for d in devices(&robot) {
        let ik = solve_ik(&d, &worlds, &goals, &IkOptions::default()).unwrap();
        assert_eq!(ik.problems.len(), goals.len());
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem {
                world: s.problem.world,
                start: robot.default_q().to_vec(),
                goal: s.solution.to_vec(),
            })
            .collect();
        assert!(problems.len() >= 2, "{}: too few IK solutions to test", d.name());
        let result = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert!(
            result.solved().any(|s| s.index as u32 != s.problem.world),
            "{}: no solved problem whose index differs from its world",
            d.name()
        );
        for s in result.solved() {
            assert_eq!(s.problem.world, problems[s.index].world);
            // Every waypoint of every reported path is collision-free in its own world.
            let n = robot.dof();
            let item_world = vec![s.problem.world; s.solution.len() / n];
            let e = cpu.evaluate(&worlds, &item_world, s.solution, &CollisionWeights::NONE).unwrap();
            assert!(
                (0..item_world.len()).all(|i| e.collision_free(i)),
                "{}: path collides in world {}",
                d.name(),
                s.problem.world
            );
        }
    }
}

#[test]
fn trajopt_smoothness_gradient_matches_its_cost() {
    // Collision off: the trajectory cost is pure smoothness,
    //   J = w_acc * sum |q[t+1] - 2 q[t] + q[t-1]|^2 + w_vel * sum |q[t+1] - q[t]|^2,
    // whose gradient one linearized optimizer step must reproduce (see tests/gpu.rs).
    let robot = panda();
    let n = robot.dof();
    let worlds = vec![World::default()];
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.0;
    goal[2] -= 0.7;
    let problems = vec![PlanProblem { world: 0, start: robot.default_q().to_vec(), goal }];
    let o = PlanOptions { collision: CollisionWeights::NONE, ..Default::default() };
    let cost = |path: &[f64]| {
        let t_count = path.len() / n;
        let q = |t: usize, j: usize| path[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..t_count - 1 {
                c += o.w_vel as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..t_count - 1 {
                c += o.w_acc as f64 * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    for d in devices(&robot) {
        let (lr, eps) = (1e3, 1e7);
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let one = PlanOptions { iterations: 1, learning_rate: lr, adam_epsilon: eps, ..o };
        let stepped = plan(&d, &worlds, &problems, &one).unwrap().paths.positions;
        // Seed 1's path (seed 0 is a straight line, where the smoothness gradient vanishes); the
        // result holds `o.seeds` paths back to back.
        let waypoints = o.waypoints;
        let offset = waypoints * n;
        let path: Vec<f64> = seed[offset..2 * offset].iter().map(|&v| v as f64).collect();
        let (mut worst, mut largest) = (0.0f64, 0.0f64);
        for i in n..(waypoints - 1) * n {
            let h = 1e-5;
            let (mut up, mut down) = (path.clone(), path.clone());
            up[i] += h;
            down[i] -= h;
            let fd = (cost(&up) - cost(&down)) / (2.0 * h);
            let from_step = (seed[offset + i] - stepped[offset + i]) as f64 * (eps / lr) as f64;
            worst = worst.max((fd - from_step).abs() / fd.abs().max(1.0));
            largest = largest.max(fd.abs());
        }
        assert!(largest > 0.1, "{}: test path is too straight to exercise the gradient ({largest})", d.name());
        assert!(worst < 1e-2, "{}: smoothness gradient off by {worst:.2e} relative", d.name());
    }
}
