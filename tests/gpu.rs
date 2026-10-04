//! The GPU device against the CPU device, plus end-to-end validity of GPU plans.
//! Skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::*;

fn setup() -> Option<(Robot, Device, Device)> {
    let robot = Robot::from_config_file(common::panda_config()).unwrap();
    match Device::gpu(&robot) {
        Ok(gpu) => {
            eprintln!("using {}", gpu.name());
            let cpu = Device::cpu(&robot);
            Some((robot, gpu, cpu))
        }
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
            eprintln!("skipping GPU test: {e}");
            None
        }
        Err(e) => panic!("no GPU: {e}"),
    }
}

#[test]
fn unknown_adapter_is_a_clear_error() {
    let robot = Robot::from_config_file(common::panda_config()).unwrap();
    let err = Device::gpu_named(&robot, "no-such-gpu").err().expect("should fail").to_string();
    assert!(err.contains("no-such-gpu"), "{err}");
}

fn worlds(n: usize, seed: u64) -> Vec<World> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| common::tabletop(&mut rng)).collect()
}

fn ik_problems(worlds: &[World], seed: u64) -> Vec<IkProblem> {
    let mut rng = Rng::new(seed);
    worlds
        .iter()
        .enumerate()
        .map(|(i, s)| IkProblem { world: i as u32, target: common::grasp_target(s, &mut rng) })
        .collect()
}

#[test]
fn evaluate_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let worlds = worlds(16, 3);
    let mut rng = Rng::new(9);
    let items = 20_000;
    let q: Vec<f32> = (0..items)
        .flat_map(|_| (0..n).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect::<Vec<_>>())
        .collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 16).collect();
    let w = CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.02, self_margin: 0.01 };
    let a = cpu.evaluate(&worlds, &item_world, &q, &w).unwrap();
    let b = gpu.evaluate(&worlds, &item_world, &q, &w).unwrap();
    let mut worst = [0.0f32; 4];
    for i in 0..items {
        worst[0] = worst[0].max((a.world_clearance[i] - b.world_clearance[i]).abs());
        worst[1] = worst[1].max((a.self_clearance[i] - b.self_clearance[i]).abs());
        worst[2] = worst[2].max((a.cost[i] - b.cost[i]).abs() / a.cost[i].abs().max(1.0));
        // Gradient error relative to the gradient's norm (components cancel in f32).
        let row = |g: &[f32]| g[i * n..(i + 1) * n].to_vec();
        let (ga, gb) = (row(&a.grad), row(&b.grad));
        let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let diff = ga.iter().zip(&gb).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        worst[3] = worst[3].max(diff / norm);
    }
    eprintln!(
        "max |cpu-gpu|: world {:.2e} self {:.2e} cost(rel) {:.2e} grad(rel) {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(worst[0] < 1e-4 && worst[1] < 1e-4 && worst[2] < 1e-3 && worst[3] < 1e-3);
}

#[test]
fn ik_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let worlds = worlds(64, 4);
    let problems = ik_problems(&worlds, 5);
    let o = IkOptions::default();
    let a = solve_ik(&cpu, &worlds, &problems, &o).unwrap();
    let b = solve_ik(&gpu, &worlds, &problems, &o).unwrap();
    let agree = a.success.iter().zip(&b.success).filter(|(x, y)| x == y).count();
    let (sa, sb) = (a.success.iter().filter(|&&s| s).count(), b.success.iter().filter(|&&s| s).count());
    eprintln!("seed successes cpu {sa} gpu {sb}, agreement {agree}/{}", a.success.len());
    assert!(agree as f32 >= 0.97 * a.success.len() as f32);
    // Every GPU success must be a real solution according to the CPU model.
    let eval = cpu
        .evaluate(
            &worlds,
            &(0..b.success.len()).map(|i| (i / o.seeds) as u32).collect::<Vec<_>>(),
            &b.q,
            &CollisionWeights::NONE,
        )
        .unwrap();
    for i in (0..b.success.len()).filter(|&i| b.success[i]) {
        let target = &problems[i / o.seeds].target;
        assert!((robot.ee_pose(b.solution(i)).position - target.position).length() < o.position_tolerance * 1.01);
        assert!(eval.collision_free(i));
    }
}

#[test]
fn gpu_plans_are_collision_free_under_cpu_check() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let worlds = worlds(128, 6);
    let problems = ik_problems(&worlds, 7);
    let ik = solve_ik(&gpu, &worlds, &problems, &IkOptions::default()).unwrap();
    let plan_problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec() })
        .collect();
    let o = PlanOptions::default();
    let result = plan(&gpu, &worlds, &plan_problems, &o).unwrap();
    let solved = result.solved().count();
    eprintln!("ik solved {}/{}; planned {solved}/{}", plan_problems.len(), problems.len(), plan_problems.len());
    assert!(solved as f32 >= 0.8 * plan_problems.len() as f32);

    // Independent check: CPU model, 4x denser interpolation than the planner's validation.
    let substeps = o.validate_substeps * 4;
    let (mut dense, mut dense_world) = (vec![], vec![]);
    for Solved { problem: prob, solution: tr, .. } in result.solved() {
        assert!(tr[..n].iter().zip(&prob.start).all(|(a, b)| a == b), "start moved");
        assert!(tr[tr.len() - n..].iter().zip(&prob.goal).all(|(a, b)| a == b), "goal moved");
        for t in 0..o.waypoints - 1 {
            for s in 0..=substeps {
                let a = s as f32 / substeps as f32;
                dense.extend((0..n).map(|j| tr[t * n + j] + a * (tr[(t + 1) * n + j] - tr[t * n + j])));
                dense_world.push(prob.world);
            }
        }
    }
    let eval = cpu.evaluate(&worlds, &dense_world, &dense, &CollisionWeights::NONE).unwrap();
    let worst = eval.world_clearance.iter().chain(&eval.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v));
    eprintln!("worst clearance along {} dense samples: {worst:.4} m", dense_world.len());
    assert!(worst > -2e-3, "trajectory penetrates by {worst}");
}

/// Trajectory-cost gradients recovered from one optimizer step. With a huge Adam epsilon a single
/// step is plain gradient descent, moving each waypoint by `-(lr / epsilon) * gradient`, so
/// `(seed - stepped) * epsilon / lr` is the gradient each device computed. Returns the gradients
/// and a mask of values a joint limit clamped (those carry no gradient information).
fn one_step_gradients(
    d: &Device,
    worlds: &[World],
    problems: &[PlanProblem],
    o: &PlanOptions,
) -> (Vec<f32>, Vec<bool>) {
    let (lr, eps) = (1e3, 1e7);
    let seed = plan(d, worlds, problems, &PlanOptions { iterations: 0, ..*o }).unwrap().paths.positions;
    let step = PlanOptions { iterations: 1, learning_rate: lr, adam_epsilon: eps, ..*o };
    let stepped = plan(d, worlds, problems, &step).unwrap().paths.positions;
    let robot = d.robot();
    let n = robot.dof();
    let clamped =
        stepped.iter().enumerate().map(|(i, &q)| q <= robot.lower()[i % n] || q >= robot.upper()[i % n]).collect();
    (seed.iter().zip(&stepped).map(|(a, b)| (a - b) * (eps / lr)).collect(), clamped)
}

#[test]
fn trajopt_gradients_match_cpu_element_wise() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let worlds = worlds(32, 8);
    let ik = solve_ik(&cpu, &worlds, &ik_problems(&worlds, 9), &IkOptions::default()).unwrap();
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec() })
        .collect();
    let o = PlanOptions::default();
    let (a, clamped_a) = one_step_gradients(&cpu, &worlds, &problems, &o);
    let (b, clamped_b) = one_step_gradients(&gpu, &worlds, &problems, &o);
    let mut worst = 0.0f32;
    for (w, (ga, gb)) in a.chunks(n).zip(b.chunks(n)).enumerate() {
        if (0..n).any(|j| clamped_a[w * n + j] || clamped_b[w * n + j]) {
            continue;
        }
        let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let diff = ga.iter().zip(gb).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        worst = worst.max(diff / norm);
    }
    eprintln!("worst per-waypoint |cpu-gpu| / |grad|: {worst:.2e}");
    assert!(worst < 1e-2, "trajectory gradients differ by {worst} of their norm");
}
