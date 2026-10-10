//! The MotionBenchMaker and MπNets Panda problem sets (2,600 problems, as packaged by robometrics).
//! Fetch them first with `scripts/fetch_benchmark.sh`.
//!
//! cargo run --release --example benchmark -- [data_dir=data/robometrics] [--latency N=20] [--device gpu|cpu|all]
//!
//! For every set and device it reports two modes:
//! - **plan**: plan from the start to the set's first IK solution (planning only);
//! - **ik+plan**: solve IK for the goal pose, then plan to the best solution.
//!
//! A problem succeeds when the final `panda_hand` position is within 1 cm of the goal, every joint
//! is within its limits, and an independent CPU check at 4x the planner's validation density finds
//! no collision. Batch-1 latency is measured on the first N problems of each set; everything else
//! runs each set as one batch.

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use batchplan::timing::{RetimeOptions, retime};
use batchplan::*;
use glam::{Affine3A, Quat, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
struct Problem {
    start: Vec<f32>,
    goal_pose: GoalPose,
    goal_ik: Vec<Vec<f32>>,
    obstacles: Obstacles,
    world_frame: String,
}

#[derive(Deserialize)]
struct GoalPose {
    frame: String,
    position_xyz: [f32; 3],
    quaternion_wxyz: [f32; 4],
}

#[derive(Deserialize)]
struct Obstacles {
    cuboid: Option<BTreeMap<String, Cuboid>>,
    cylinder: Option<BTreeMap<String, Cylinder>>,
}

#[derive(Deserialize)]
struct Cuboid {
    dims: [f32; 3],
    pose: [f32; 7],
}

#[derive(Deserialize)]
struct Cylinder {
    height: f32,
    radius: f32,
    pose: [f32; 7],
}

/// `[x, y, z, qw, qx, qy, qz]`.
fn pose(p: &[f32; 7]) -> (Vec3, Quat) {
    (Vec3::new(p[0], p[1], p[2]), Quat::from_xyzw(p[4], p[5], p[6], p[3]).normalize())
}

impl Problem {
    fn world(&self) -> World {
        let mut obstacles = vec![];
        for c in self.obstacles.cuboid.iter().flat_map(|m| m.values()) {
            let (center, rotation) = pose(&c.pose);
            obstacles.push(Obstacle::Cuboid { center, half_extents: Vec3::from(c.dims) * 0.5, rotation });
        }
        for c in self.obstacles.cylinder.iter().flat_map(|m| m.values()) {
            let (center, rotation) = pose(&c.pose);
            obstacles.push(Obstacle::Cylinder { center, rotation, radius: c.radius, half_height: c.height * 0.5 });
        }
        World { obstacles }
    }

    fn hand_goal(&self) -> Pose {
        let [w, x, y, z] = self.goal_pose.quaternion_wxyz;
        Pose { position: Vec3::from(self.goal_pose.position_xyz), rotation: Quat::from_xyzw(x, y, z, w).normalize() }
    }
}

fn load(path: &PathBuf) -> Result<BTreeMap<String, Vec<Problem>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {} (run scripts/fetch_benchmark.sh first)", path.display()))?;
    let sets: BTreeMap<String, Vec<Problem>> =
        // The files are large but checksummed by the fetch script, so the parser's size budget is off.
        serde_saphyr::from_str_with_options(&text, serde_saphyr::options! { budget: None })
            .with_context(|| format!("parsing {}", path.display()))?;
    for p in sets.values().flatten() {
        ensure!(p.goal_pose.frame == "panda_hand" && p.world_frame == "panda_link0", "unexpected frames");
    }
    Ok(sets)
}

struct Args {
    data: PathBuf,
    latency: usize,
    device: String,
}

fn args() -> Result<Args> {
    let mut a = Args { data: PathBuf::from("data/robometrics"), latency: 20, device: "all".into() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--latency" => a.latency = it.next().context("--latency N")?.parse()?,
            "--device" => a.device = it.next().context("--device gpu|cpu|all")?,
            s if s.starts_with("--") => bail!("unknown flag {s}"),
            s => a.data = PathBuf::from(s),
        }
    }
    Ok(a)
}

/// Converts `panda_hand` goals into goals for the robot's IK frame.
struct Frames {
    hand_to_ee: Affine3A,
}

impl Frames {
    fn new(robot: &Robot) -> Self {
        let q = robot.default_q();
        let affine = |p: Pose| Affine3A::from_rotation_translation(p.rotation, p.position);
        let hand = affine(robot.link_pose(q, "panda_hand").expect("the Panda has a panda_hand link"));
        Self { hand_to_ee: hand.inverse() * affine(robot.ee_pose(q)) }
    }

    fn ee_goal(&self, hand: Pose) -> Pose {
        let (_, rotation, position) = (Affine3A::from_rotation_translation(hand.rotation, hand.position)
            * self.hand_to_ee)
            .to_scale_rotation_translation();
        Pose { position, rotation }
    }
}

#[derive(Default)]
struct Outcome {
    success: usize,
    seconds: f64,
    path_length: Vec<f32>,
    motion_time: Vec<f32>,
    peak_acceleration: Vec<f32>,
    peak_jerk: Vec<f32>,
}

/// Successful problems under the independent check, plus path metrics of the successful paths.
fn score(robot: &Robot, cpu: &Device, worlds: &[World], problems: &[Problem], result: &PlanResult) -> Result<Outcome> {
    let n = robot.dof();
    let substeps = PlanOptions::default().validate_substeps * 4;
    let mut out = Outcome::default();
    for s in result.solved() {
        let path = s.solution;
        let waypoints = path.len() / n;
        let mut dense = vec![];
        for t in 0..waypoints - 1 {
            for k in 0..substeps {
                let a = k as f32 / substeps as f32;
                dense.extend((0..n).map(|j| path[t * n + j] + a * (path[(t + 1) * n + j] - path[t * n + j])));
            }
        }
        dense.extend_from_slice(&path[(waypoints - 1) * n..]);
        let items = dense.len() / n;
        let eval = cpu.evaluate(worlds, &vec![s.problem.world; items], &dense, &CollisionWeights::NONE)?;
        let collision_free = (0..items).all(|i| eval.collision_free(i));
        let within_limits =
            path.iter().enumerate().all(|(i, &q)| q >= robot.lower()[i % n] && q <= robot.upper()[i % n]);
        let goal = problems[s.problem.world as usize].hand_goal();
        let hand = robot.link_pose(&path[(waypoints - 1) * n..], "panda_hand").expect("panda_hand");
        let reached = (hand.position - goal.position).length() < 0.01;
        if !(collision_free && within_limits && reached) {
            continue;
        }
        out.success += 1;
        out.path_length.push(
            (0..waypoints - 1)
                .map(|t| (0..n).map(|j| (path[(t + 1) * n + j] - path[t * n + j]).powi(2)).sum::<f32>().sqrt())
                .sum(),
        );
        let dt = 0.01;
        let traj = retime(robot, path, &RetimeOptions { dt, ..Default::default() });
        out.motion_time.push(traj.duration);
        let v = &traj.velocities;
        let acc: Vec<f32> = (0..v.len().saturating_sub(n)).map(|i| (v[i + n] - v[i]) / dt).collect();
        let jerk: Vec<f32> = (0..acc.len().saturating_sub(n)).map(|i| (acc[i + n] - acc[i]) / dt).collect();
        out.peak_acceleration.push(acc.iter().fold(0.0, |m: f32, a| m.max(a.abs())));
        out.peak_jerk.push(jerk.iter().fold(0.0, |m: f32, a| m.max(a.abs())));
    }
    Ok(out)
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn ik_problems(problems: &[Problem], frames: &Frames) -> Vec<IkProblem> {
    problems
        .iter()
        .enumerate()
        .map(|(i, p)| IkProblem { world: i as u32, target: frames.ee_goal(p.hand_goal()) })
        .collect()
}

/// IK for every goal, then a plan from each start to its best IK solution.
fn ik_and_plan(device: &Device, worlds: &[World], problems: &[Problem], frames: &Frames) -> Result<PlanResult> {
    let ik = solve_ik(device, worlds, &ik_problems(problems, frames), &IkOptions::default())?;
    let plans: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: problems[s.problem.world as usize].start.clone(),
            goal: s.solution.to_vec(),
        })
        .collect();
    plan(device, worlds, &plans, &PlanOptions::default())
}

fn main() -> Result<()> {
    let args = args()?;
    let robot = Robot::from_config_file(common::panda_config())?;
    let frames = Frames::new(&robot);
    let cpu = Device::cpu(&robot);
    let mut devices = vec![];
    if args.device != "cpu" {
        devices.push(Device::gpu(&robot)?);
    }
    if args.device != "gpu" {
        devices.push(Device::cpu(&robot));
    }
    let files = ["mb_set.yaml", "mpinets_set.yaml"];
    let mut sets = vec![];
    for f in files {
        sets.extend(load(&args.data.join(f))?);
    }
    println!(
        "{} sets, {} problems; batch-1 latency on the first {} problems of each set\n",
        sets.len(),
        sets.iter().map(|(_, p)| p.len()).sum::<usize>(),
        args.latency
    );

    for device in &devices {
        println!("{}", device.name());
        println!(
            "{:<28} {:>5} {:>10} {:>11} {:>11} {:>9} {:>26} {:>7} {:>7} {:>8} {:>9}",
            "set",
            "n",
            "free ends",
            "plan ok",
            "ik+plan ok",
            "prob/s",
            "ik+plan batch-1 ms m/75/98",
            "length",
            "time s",
            "acc",
            "jerk"
        );
        let (mut total, mut total_free, mut total_plan, mut total_full, mut total_time) = (0, 0, 0, 0, 0.0);
        let mut all_latency = vec![];
        solve_ik(
            device,
            &[World::default()],
            &[IkProblem { world: 0, target: robot.ee_pose(robot.default_q()) }],
            &IkOptions::default(),
        )?;
        for (name, problems) in &sets {
            let worlds: Vec<World> = problems.iter().map(Problem::world).collect();
            let to_ik: Vec<PlanProblem> = problems
                .iter()
                .enumerate()
                .map(|(i, p)| PlanProblem { world: i as u32, start: p.start.clone(), goal: p.goal_ik[0].clone() })
                .collect();
            let t = Instant::now();
            let planned = plan(device, &worlds, &to_ik, &PlanOptions::default())?;
            let plan_only =
                Outcome { seconds: t.elapsed().as_secs_f64(), ..score(&robot, &cpu, &worlds, problems, &planned)? };

            let t = Instant::now();
            let full_result = ik_and_plan(device, &worlds, problems, &frames)?;
            let full =
                Outcome { seconds: t.elapsed().as_secs_f64(), ..score(&robot, &cpu, &worlds, problems, &full_result)? };

            // Problems whose start and given IK goal are collision-free under this robot model.
            let ends: Vec<f32> = problems.iter().flat_map(|p| p.start.iter().chain(&p.goal_ik[0]).copied()).collect();
            let end_world: Vec<u32> = (0..problems.len() as u32).flat_map(|i| [i, i]).collect();
            let eval = cpu.evaluate(&worlds, &end_world, &ends, &CollisionWeights::NONE)?;
            let free_ends =
                (0..problems.len()).filter(|&i| eval.collision_free(2 * i) && eval.collision_free(2 * i + 1)).count();
            total_free += free_ends;

            let mut latency = vec![];
            for i in 0..args.latency.min(problems.len()) {
                let t = Instant::now();
                ik_and_plan(device, &worlds[i..i + 1], &problems[i..i + 1], &frames)?;
                latency.push(t.elapsed().as_secs_f64() * 1e3);
            }
            all_latency.extend(&latency);
            let mean = latency.iter().sum::<f64>() / latency.len().max(1) as f64;
            let n = problems.len();
            println!(
                "{:<28} {:>5} {:>9.1}% {:>10.1}% {:>10.1}% {:>9.1} {:>10.0} {:>7.0} {:>7.0} {:>7.2} {:>7.2} {:>8.1} {:>9.0}",
                name,
                n,
                100.0 * free_ends as f64 / n as f64,
                100.0 * plan_only.success as f64 / n as f64,
                100.0 * full.success as f64 / n as f64,
                n as f64 / full.seconds,
                mean,
                percentile(&mut latency, 0.75),
                percentile(&mut latency, 0.98),
                median(&full.path_length),
                median(&full.motion_time),
                median(&full.peak_acceleration),
                median(&full.peak_jerk),
            );
            total += n;
            total_plan += plan_only.success;
            total_full += full.success;
            total_time += full.seconds;
        }
        let mean = all_latency.iter().sum::<f64>() / all_latency.len().max(1) as f64;
        println!(
            "{:<28} {:>5} {:>9.1}% {:>10.1}% {:>10.1}% {:>9.1} {:>10.0} {:>7.0} {:>7.0}\n",
            "all",
            total,
            100.0 * total_free as f64 / total as f64,
            100.0 * total_plan as f64 / total as f64,
            100.0 * total_full as f64 / total as f64,
            total as f64 / total_time,
            mean,
            percentile(&mut all_latency, 0.75),
            percentile(&mut all_latency, 0.98),
        );
    }
    Ok(())
}
