//! Behaviour every `Device` must share, checked on each available device (the GPU is skipped when
//! absent unless `BATCHPLAN_REQUIRE_GPU=1`).

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::*;
use glam::Vec3;

fn panda() -> Robot {
    common::panda().unwrap()
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
            // Every reported path is collision-free in its own world along its whole length.
            let n = robot.dof();
            let trajectory = Trajectory::new(&robot, s.solution, 1.0);
            let q = trajectory.sample(32.0 / trajectory.knot_interval).positions;
            let item_world = vec![s.problem.world; q.len() / n];
            let e = cpu.evaluate(&worlds, &item_world, &q, &CollisionWeights::NONE).unwrap();
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
        // result holds `o.seeds` paths back to back. Only control points 3..points - 3 move.
        let points = o.control_points;
        let offset = points * n;
        let path: Vec<f64> = seed[offset..2 * offset].iter().map(|&v| v as f64).collect();
        let (mut worst, mut largest) = (0.0f64, 0.0f64);
        for i in 3 * n..(points - 3) * n {
            let h = 1e-5;
            let (mut up, mut down) = (path.clone(), path.clone());
            up[i] += h;
            down[i] -= h;
            let fd = (cost(&up) - cost(&down)) / (2.0 * h);
            let from_step = (seed[offset + i] - stepped[offset + i]) as f64 * (eps / lr) as f64;
            let err = (fd - from_step).abs() / fd.abs().max(1.0);
            if err > 0.02 {
                eprintln!("  coord {i} (point {}, joint {}): fd {fd:.3} step {from_step:.3}", i / n, i % n);
            }
            worst = worst.max(err);
            largest = largest.max(fd.abs());
        }
        assert!(largest > 0.1, "{}: test path is too straight to exercise the gradient ({largest})", d.name());
        assert!(worst < 1e-2, "{}: smoothness gradient off by {worst:.2e} relative", d.name());
    }
}

/// Weights of a uniform cubic B-spline span's control points at `u` (same as the planner's).
fn basis(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        v * v * v / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u * u * u / 6.0,
    ]
}

#[test]
fn trajopt_collision_gradient_matches_its_cost() {
    // The full trajectory cost: smoothness on the control points plus the collision cost at
    // `samples_per_span` points of every span, at u = (s + 0.5) / samples_per_span.
    let robot = panda();
    let n = robot.dof();
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.2;
    let start = robot.default_q().to_vec();
    // A post between start and goal that the straight seed sweeps through.
    let middle: Vec<f32> = start.iter().zip(&goal).map(|(a, b)| 0.5 * (a + b)).collect();
    let center = robot.ee_pose(&middle).position;
    let post = Obstacle::Cylinder { center, rotation: glam::Quat::IDENTITY, radius: 0.05, half_height: 0.3 };
    let worlds = vec![World { obstacles: vec![post] }];
    let problems = vec![PlanProblem { world: 0, start, goal }];
    let o = PlanOptions::default();
    let (points, k) = (o.control_points, o.samples_per_span);
    let samples_of = |cp: &[f64]| -> Vec<f32> {
        let mut out = vec![];
        for span in 0..points - 3 {
            for s in 0..k {
                let w = basis((s as f64 + 0.5) / k as f64);
                out.extend((0..n).map(|j| (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f64>() as f32));
            }
        }
        out
    };
    let smoothness = |cp: &[f64]| {
        let q = |t: usize, j: usize| cp[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..points - 1 {
                c += o.w_vel as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..points - 1 {
                c += o.w_acc as f64 * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    let cpu = Device::cpu(&robot);
    for d in devices(&robot) {
        let (lr, eps) = (1e3, 1e7);
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let one = PlanOptions { iterations: 1, learning_rate: lr, adam_epsilon: eps, ..o };
        let stepped = plan(&d, &worlds, &problems, &one).unwrap().paths.positions;
        // Seed 0, the straight line through the post.
        let path: Vec<f64> = seed[..points * n].iter().map(|&v| v as f64).collect();
        let free: Vec<usize> = (3 * n..(points - 3) * n).collect();
        // Central differences at two step sizes: the cost is only piecewise smooth (obstacle
        // edges, hinge activation) and evaluated in f32, so take the closer of the two.
        let fd_at = |h: f64| -> Vec<f64> {
            let mut perturbed = vec![];
            for &i in &free {
                for sign in [1.0, -1.0] {
                    let mut p = path.clone();
                    p[i] += sign * h;
                    perturbed.extend(samples_of(&p));
                }
            }
            let per_path = (points - 3) * k;
            let items = perturbed.len() / n;
            let costs = cpu.evaluate(&worlds, &vec![0; items], &perturbed, &o.collision).unwrap().cost;
            let collision = |p: usize| costs[p * per_path..(p + 1) * per_path].iter().map(|&c| c as f64).sum::<f64>();
            free.iter()
                .enumerate()
                .map(|(f, &i)| {
                    let (mut up, mut down) = (path.clone(), path.clone());
                    up[i] += h;
                    down[i] -= h;
                    (collision(2 * f) - collision(2 * f + 1) + smoothness(&up) - smoothness(&down)) / (2.0 * h)
                })
                .collect()
        };
        let (fine, coarse) = (fd_at(2e-4), fd_at(1e-3));
        let largest = fine.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        for (f, &i) in free.iter().enumerate() {
            let from_step = (seed[i] - stepped[i]) as f64 * (eps / lr) as f64;
            let scale = fine[f].abs().max(0.05 * largest);
            let err = (fine[f] - from_step).abs().min((coarse[f] - from_step).abs()) / scale;
            worst = worst.max(err);
        }
        eprintln!("{}: largest gradient {largest:.1}, worst relative error {worst:.2e}", d.name());
        let seed_samples = samples_of(&path);
        let items = seed_samples.len() / n;
        let touching = cpu.evaluate(&worlds, &vec![0; items], &seed_samples, &o.collision).unwrap().cost;
        assert!(touching.iter().filter(|&&c| c > 1.0).count() > 3, "{}: the straight seed misses the post", d.name());
        assert!(worst < 2e-2, "{}: trajectory gradient off by {worst:.2e} relative", d.name());
    }
}
