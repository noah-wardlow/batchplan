//! Generates policy-training demonstrations: nominal reach-to-grasp trajectories in random
//! cluttered worlds plus recovery trajectories from perturbed mid-path states, retimed with a
//! minimum-jerk profile and randomized speed, written as NumPy `.npy` files.
//!
//! cargo run --release --example datagen -- <out_dir> [episodes=256] [dt=0.05]

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::datagen::{RecoveryOptions, recovery_problems};
use batchplan::npy::write_npy;
use batchplan::rng::Rng;
use batchplan::timing::{RetimeOptions, retime};
use batchplan::*;

struct Episode {
    /// 0 = nominal, 1 = recovery.
    kind: u8,
    /// Row of the nominal episode a recovery branches from (-1 for nominal episodes).
    parent: i32,
    world: u32,
    goal_pose: Pose,
    trajectory: JointTrajectory,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().context("usage: datagen <out_dir> [episodes] [dt]")?);
    let episodes: usize = args.next().map_or(Ok(256), |a| a.parse())?;
    let dt: f32 = args.next().map_or(Ok(0.05), |a| a.parse())?;
    std::fs::create_dir_all(&out)?;

    let robot = Robot::from_config_file(common::panda_config())?;
    let n = robot.dof();
    let device = Device::gpu(&robot).unwrap_or_else(|e| {
        eprintln!("no GPU ({e}); using the CPU");
        Device::cpu(&robot)
    });
    println!("device: {}", device.name());
    let started = Instant::now();
    let mut rng = Rng::new(7);

    let worlds: Vec<World> = (0..episodes).map(|_| common::tabletop(&mut rng)).collect();
    let targets: Vec<Pose> = worlds.iter().map(|w| common::grasp_target(w, &mut rng)).collect();
    let ik_problems: Vec<IkProblem> =
        targets.iter().enumerate().map(|(i, &target)| IkProblem { world: i as u32, target }).collect();
    let ik = solve_ik(&device, &worlds, &ik_problems, &IkOptions::default())?;

    // Random collision-free starts around the default pose, for start-state diversity.
    let mut starts: Vec<f32> = (0..episodes)
        .flat_map(|_| {
            (0..n)
                .map(|j| (robot.default_q()[j] + 0.3 * rng.normal()).clamp(robot.lower()[j], robot.upper()[j]))
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = (0..episodes as u32).collect();
    let start_eval = device.evaluate(&worlds, &item_world, &starts, &CollisionWeights::NONE)?;
    for e in (0..episodes).filter(|&e| !start_eval.collision_free(e)) {
        starts[e * n..(e + 1) * n].copy_from_slice(robot.default_q());
    }
    let (goal_poses, problems): (Vec<Pose>, Vec<PlanProblem>) = (0..episodes)
        .filter_map(|e| {
            let goal = ik.best(e)?.to_vec();
            Some((targets[e], PlanProblem { world: e as u32, start: starts[e * n..(e + 1) * n].to_vec(), goal }))
        })
        .unzip();
    let plan_opts = PlanOptions::default();
    let nominal = plan(&device, &worlds, &problems, &plan_opts)?;
    let recoveries = recovery_problems(&device, &worlds, &problems, &nominal, &RecoveryOptions::default())?;
    let recovery_plans: Vec<PlanProblem> = recoveries.iter().map(|r| r.problem.clone()).collect();
    let recovered = plan(&device, &worlds, &recovery_plans, &plan_opts)?;
    let planning_time = started.elapsed().as_secs_f64();

    let mut retimed = |path: &[f32]| {
        retime(&robot, path, &RetimeOptions { speed_scale: rng.range(0.6, 1.0), dt, ..Default::default() })
    };
    let mut rows: Vec<Episode> = vec![];
    let mut row_of_problem = vec![-1i32; problems.len()];
    for (p, problem) in problems.iter().enumerate() {
        if let Some(path) = nominal.best(p) {
            row_of_problem[p] = rows.len() as i32;
            let trajectory = retimed(path);
            rows.push(Episode { kind: 0, parent: -1, world: problem.world, goal_pose: goal_poses[p], trajectory });
        }
    }
    for (r, rec) in recoveries.iter().enumerate() {
        if let Some(path) = recovered.best(r) {
            let trajectory = retimed(path);
            let parent = row_of_problem[rec.parent];
            rows.push(Episode {
                kind: 1,
                parent,
                world: rec.problem.world,
                goal_pose: goal_poses[rec.parent],
                trajectory,
            });
        }
    }

    // Pad to a common length: positions hold their final value, velocities are zero.
    let m = rows.len();
    let horizon = rows.iter().map(|e| e.trajectory.len()).max().unwrap_or(0);
    let mut positions = vec![0.0f32; m * horizon * n];
    let mut velocities = vec![0.0f32; m * horizon * n];
    for (i, e) in rows.iter().enumerate() {
        let t = &e.trajectory;
        for h in 0..horizon {
            let src = h.min(t.len() - 1) * n;
            let dst = (i * horizon + h) * n;
            positions[dst..dst + n].copy_from_slice(&t.positions[src..src + n]);
            if h < t.len() {
                velocities[dst..dst + n].copy_from_slice(&t.velocities[src..src + n]);
            }
        }
    }
    let column = |f: &dyn Fn(&Episode) -> i32| rows.iter().map(f).collect::<Vec<i32>>();
    write_npy(out.join("positions.npy"), &[m, horizon, n], &positions)?;
    write_npy(out.join("velocities.npy"), &[m, horizon, n], &velocities)?;
    write_npy(out.join("length.npy"), &[m], &column(&|e| e.trajectory.len() as i32))?;
    write_npy(out.join("parent.npy"), &[m], &column(&|e| e.parent))?;
    write_npy(out.join("world.npy"), &[m], &column(&|e| e.world as i32))?;
    let kinds: Vec<u8> = rows.iter().map(|e| e.kind).collect();
    write_npy(out.join("kind.npy"), &[m], &kinds)?;
    let poses: Vec<f32> = rows
        .iter()
        .flat_map(|e| {
            let (p, q) = (e.goal_pose.position, e.goal_pose.rotation);
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        })
        .collect();
    write_npy(out.join("goal_pose.npy"), &[m, 7], &poses)?;
    std::fs::write(out.join("worlds.json"), serde_json::to_string(&worlds)?)?;
    let recovery_rows = kinds.iter().filter(|&&k| k == 1).count();
    let meta = serde_json::json!({
        "dt": dt,
        "joint_names": robot.joint_names(),
        "max_velocity": robot.max_velocity(),
        "max_acceleration": RetimeOptions::default().max_acceleration,
        "episodes": m,
        "nominal": m - recovery_rows,
        "recovery": recovery_rows,
        "kind": {"0": "nominal", "1": "recovery from a perturbed state of episode `parent`"},
        "goal_pose": "ee frame target: position xyz + quaternion xyzw",
        "device": device.name(),
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    println!(
        "{episodes} worlds -> IK solved {}, nominal planned {}, recovery starts {} -> planned {recovery_rows}",
        problems.len(),
        m - recovery_rows,
        recoveries.len(),
    );
    println!(
        "wrote {m} episodes x {horizon} steps (dt {dt} s) to {}; planning took {planning_time:.2} s",
        out.display()
    );
    Ok(())
}
