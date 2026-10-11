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
    let mut devices = vec![Device::cpu(robot).unwrap()];
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
    let scene = vec![World { obstacles: vec![Obstacle::Sphere { center: Vec3::new(2.0, 0.0, 0.0), radius: 0.1 }] }];
    let none = CollisionWeights::NONE;
    for d in devices(&robot) {
        let name = d.name();
        let worlds = d.upload(&scene).unwrap();
        let elsewhere = Device::cpu(&robot).unwrap().upload(&scene).unwrap();
        let err = d.evaluate(&elsewhere, &[0], &q, &none).unwrap_err();
        assert!(matches!(err, Error::Input(_)), "{name}: worlds uploaded to another device give {err:?}");
        assert!(d.evaluate(&worlds, &[0, 0], &q, &none).is_err(), "{name}: two items, one configuration");
        assert!(d.evaluate(&worlds, &[0], &q[..6], &none).is_err(), "{name}: configuration missing a joint");
        assert!(d.evaluate(&worlds, &[1], &q, &none).is_err(), "{name}: world index past the end");
        let target = robot.ee_pose(&q);
        assert!(solve_ik(&d, &worlds, &[IkProblem { world: 1, target }], &IkOptions::default()).is_err(), "{name}: IK");
        let problem = PlanProblem { world: 1, start: q.clone(), goal: q.clone() };
        assert!(plan(&d, &worlds, &[problem], &PlanOptions::default()).is_err(), "{name}: plan");
        // Inputs no device could check consistently are refused before any work.
        let input = |r: Result<_, Error>| matches!(r, Err(Error::Input(_)));
        let mut beyond = q.clone();
        beyond[3] = robot.upper()[3] + 0.5;
        let past_limits = PlanProblem { world: 0, start: q.clone(), goal: beyond };
        assert!(
            input(plan(&d, &worlds, &[past_limits], &PlanOptions::default()).map(|_| ())),
            "{name}: goal past limits"
        );
        let skewed = Pose { rotation: glam::Quat::from_xyzw(0.0, 0.0, 0.0, 2.0), ..target };
        let ik = solve_ik(&d, &worlds, &[IkProblem { world: 0, target: skewed }], &IkOptions::default());
        assert!(input(ik.map(|_| ())), "{name}: a non-unit target rotation");
        let rotated = |rotation| Obstacle::Cuboid { center: Vec3::ZERO, half_extents: Vec3::ONE, rotation };
        for bad in [
            Obstacle::Sphere { center: Vec3::new(f32::NAN, 0.0, 0.0), radius: 0.1 },
            Obstacle::Sphere { center: Vec3::ZERO, radius: -0.1 },
            rotated(glam::Quat::from_xyzw(0.0, 0.0, 0.0, 0.5)),
        ] {
            let err = d.upload(&[World { obstacles: vec![bad.clone()] }]).map(|_| ());
            assert!(input(err), "{name}: {bad:?} uploaded");
        }
    }
}

#[test]
fn clearances_are_exact_up_to_the_margins() {
    let robot = panda();
    let (n, items) = (robot.dof(), 4000);
    let mut rng = batchplan::rng::Rng::new(5);
    let q: Vec<f32> = (0..items * n).map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n])).collect();
    let item_world = vec![0; items];
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let margin = 0.05;
    let near = CollisionWeights { world: 1.0, self_collision: 1.0, margin, self_margin: margin };
    // Margins wider than the robot leave nothing far enough apart to skip.
    let everything = CollisionWeights { margin: 10.0, self_margin: 10.0, ..CollisionWeights::NONE };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let exact = d.evaluate(&worlds, &item_world, &q, &everything).unwrap();
        let gated = d.evaluate(&worlds, &item_world, &q, &near).unwrap();
        let kinds = [
            ("world", exact.world_clearance, gated.world_clearance),
            ("self", exact.self_clearance, gated.self_clearance),
        ];
        for (kind, exact, gated) in kinds {
            let (mut close, mut bounded) = (0, 0);
            for (&e, &g) in exact.iter().zip(&gated) {
                if e <= margin {
                    assert_eq!(g, e, "{}: a close {kind} pair was skipped", d.name());
                    close += 1;
                } else {
                    assert!(g > margin && g <= e + 1e-5, "{}: {kind} {g} does not bound {e}", d.name());
                    bounded += usize::from(g < e);
                }
            }
            assert!(close > 100 && bounded > 20, "{}: {kind}: {close} close, {bounded} bounded", d.name());
        }
    }
}

#[test]
fn ik_reports_the_clearances_evaluate_finds() {
    let robot = panda();
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let mut rng = batchplan::rng::Rng::new(9);
    let problems: Vec<IkProblem> =
        (0..16).map(|_| IkProblem { world: 0, target: common::grasp_target(&scene[0], &mut rng) }).collect();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = solve_ik(&d, &worlds, &problems, &IkOptions::default()).unwrap();
        let items = ik.q.len() / robot.dof();
        let e = d.evaluate(&worlds, &vec![0; items], &ik.q, &CollisionWeights::NONE).unwrap();
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6);
        assert!(close(&ik.world_clearance, &e.world_clearance), "{}: world clearance", d.name());
        assert!(close(&ik.self_clearance, &e.self_clearance), "{}: self clearance", d.name());
        assert!(!close(&ik.world_clearance, &ik.self_clearance), "world and self clearance should differ");
    }
}

#[test]
fn evaluation_does_not_depend_on_the_batch() {
    // Small batches split each configuration's collision work across several GPU invocations,
    // large ones give it one; the results must agree.
    let robot = panda();
    let n = robot.dof();
    let mut rng = batchplan::rng::Rng::new(13);
    let q: Vec<f32> = (0..40_000 * n).map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n])).collect();
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let w = CollisionWeights { world: 1.0, self_collision: 1.0, margin: 0.05, self_margin: 0.02 };
    let few = 64;
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let alone = d.evaluate(&worlds, &vec![0; few], &q[..few * n], &w).unwrap();
        let batched = d.evaluate(&worlds, &vec![0; q.len() / n], &q, &w).unwrap();
        assert!(alone.cost.iter().filter(|&&c| c > 0.0).count() > 10, "too few configurations carry cost");
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0));
        assert!(close(&alone.world_clearance, &batched.world_clearance[..few]), "{}: world clearance", d.name());
        assert!(close(&alone.self_clearance, &batched.self_clearance[..few]), "{}: self clearance", d.name());
        assert!(close(&alone.cost, &batched.cost[..few]), "{}: cost", d.name());
        assert!(close(&alone.grad, &batched.grad[..few * n]), "{}: gradient", d.name());
    }
}

#[test]
fn obstacle_free_worlds_work_on_every_device() {
    let robot = panda();
    let start = robot.default_q().to_vec();
    let mut elsewhere = start.clone();
    elsewhere[0] += 0.8;
    elsewhere[3] += 0.4;
    let target = robot.ee_pose(&elsewhere);
    for d in devices(&robot) {
        let name = d.name();
        let worlds = d.upload(&[World::default()]).unwrap();
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
    let scene: Vec<World> = (0..2).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = [1u32, 1, 0]
        .iter()
        .map(|&w| IkProblem { world: w, target: common::grasp_target(&scene[w as usize], &mut rng) })
        .collect();
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
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
            let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
            let q = trajectory.sample(32.0 / trajectory.knot_interval).unwrap().positions;
            let item_world = vec![s.problem.world; q.len() / n];
            let e = cpu.evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE).unwrap();
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
    // whose gradient sets the direction of the first L-BFGS step (see tests/gpu.rs).
    let robot = panda();
    let n = robot.dof();
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.0;
    goal[2] -= 0.7;
    let problems = vec![PlanProblem { world: 0, start: robot.default_q().to_vec(), goal }];
    let scene = vec![World::default()];
    let o = PlanOptions { collision: CollisionWeights::NONE, fallback: None, ..Default::default() };
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
        let worlds = d.upload(&scene).unwrap();
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let stepped = plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o }).unwrap().paths.positions;
        // Seed 1's path (seed 0 is a straight line, where the smoothness gradient vanishes); the
        // result holds `o.seeds` paths back to back. Only control points 3..points - 3 move.
        let points = o.control_points;
        let offset = points * n;
        let path: Vec<f64> = seed[offset..2 * offset].iter().map(|&v| v as f64).collect();
        let free = 3 * n..(points - 3) * n;
        let fd: Vec<f64> = free
            .clone()
            .map(|i| {
                let h = 1e-5;
                let (mut up, mut down) = (path.clone(), path.clone());
                up[i] += h;
                down[i] -= h;
                (cost(&up) - cost(&down)) / (2.0 * h)
            })
            .collect();
        // Without history the step is steepest descent: -gradient times a positive scale.
        let step: Vec<f64> = free.map(|i| (seed[offset + i] - stepped[offset + i]) as f64).collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fd_max, step_max) = (largest(&fd), largest(&step));
        assert!(fd_max > 0.1, "{}: test path is too straight to exercise the gradient ({fd_max})", d.name());
        assert!(step_max > 0.0, "{}: the step did not move the path", d.name());
        let worst = fd.iter().zip(&step).map(|(f, s)| (f / fd_max - s / step_max).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-2, "{}: the step's direction is off the smoothness gradient by {worst:.2e}", d.name());
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
    let scene = vec![World { obstacles: vec![post] }];
    let problems = vec![PlanProblem { world: 0, start, goal }];
    let o = PlanOptions { fallback: None, ..Default::default() };
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
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let stepped = plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o }).unwrap().paths.positions;
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
            let costs = cpu.evaluate(&on_cpu, &vec![0; items], &perturbed, &o.collision).unwrap().cost;
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
        // Without history the step is steepest descent: -gradient times a positive scale, so
        // gradients are compared scaled to their largest entries.
        let step: Vec<f64> = free.iter().map(|&i| (seed[i] - stepped[i]) as f64).collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fine_max, coarse_max, step_max) = (largest(&fine), largest(&coarse), largest(&step));
        assert!(step_max > 0.0, "{}: the step did not move the path", d.name());
        let mut worst = 0.0f64;
        for f in 0..free.len() {
            let s = step[f] / step_max;
            let err = (fine[f] / fine_max - s).abs().min((coarse[f] / coarse_max - s).abs());
            worst = worst.max(err / (fine[f].abs() / fine_max).max(0.05));
        }
        eprintln!("{}: largest gradient {fine_max:.1}, worst relative error {worst:.2e}", d.name());
        let seed_samples = samples_of(&path);
        let items = seed_samples.len() / n;
        let touching = cpu.evaluate(&on_cpu, &vec![0; items], &seed_samples, &o.collision).unwrap().cost;
        assert!(touching.iter().filter(|&&c| c > 1.0).count() > 3, "{}: the straight seed misses the post", d.name());
        assert!(worst < 2e-2, "{}: trajectory gradient off by {worst:.2e} relative", d.name());
    }
}

/// 64 tabletop reaches from the default pose, the worlds they live in, and their IK goals.
fn reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = batchplan::rng::Rng::new(41);
    let scene: Vec<World> = (0..64).map(|_| common::tabletop(&mut rng)).collect();
    let cpu = Device::cpu(robot).unwrap();
    let goals: Vec<IkProblem> = scene
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng) })
        .collect();
    let ik = solve_ik(&cpu, &cpu.upload(&scene).unwrap(), &goals, &IkOptions::default()).unwrap();
    let problems = ik
        .solved()
        .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec() })
        .collect();
    (scene, problems)
}

#[test]
fn a_tiny_time_budget_returns_after_at_most_one_more_chunk() {
    let robot = panda();
    let (scene, problems) = reaches(&robot);
    // Enough rounds (20 chunks) that a fast GPU's plan still lasts well beyond one chunk.
    let o = PlanOptions { iterations: 160, ..Default::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let timed = |o: &PlanOptions| {
            let t = std::time::Instant::now();
            let result = plan(&d, &worlds, &problems, o).unwrap();
            (t.elapsed(), result)
        };
        timed(&o); // warm up
        let (full, _) = timed(&o);
        // A GPU checks the budget between submissions of 8 rounds; time one such chunk, with
        // seeding and validation.
        let (chunk, _) = timed(&PlanOptions { iterations: 8, fallback: None, ..o });
        let budget = std::time::Duration::from_millis(1);
        let (spent, result) = timed(&PlanOptions { time_budget: Some(budget), ..o });
        eprintln!("{}: full {full:?}, one chunk {chunk:?}, budget {budget:?} -> {spent:?}", d.name());
        assert!(spent < budget + 2 * chunk + std::time::Duration::from_millis(20), "{}: took {spent:?}", d.name());
        assert!(spent < full / 2, "{}: the budget saved little ({spent:?} of {full:?})", d.name());
        assert_eq!(result.valid.len(), problems.len() * o.seeds);
    }
}

#[test]
fn planning_without_a_budget_gives_the_same_result_every_run() {
    let robot = panda();
    let (scene, problems) = reaches(&robot);
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let a = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        let b = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert_eq!(a.paths, b.paths, "{}: paths differ between runs", d.name());
        assert_eq!((a.valid, a.length), (b.valid, b.length), "{}", d.name());
    }
}
