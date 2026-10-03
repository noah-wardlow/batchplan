//! Generates policy-training demonstrations: reach-to-grasp trajectories in random cluttered
//! worlds plus recoveries from perturbed mid-path states, retimed with a minimum-jerk profile and
//! randomized speed.
//!
//! cargo run --release --example datagen -- <out_dir> [episodes=256] [fps=20]
//!     writes plain NumPy `.npy` arrays
//! cargo run --release --features lerobot --example datagen -- --lerobot <out_dir> [episodes] [fps]
//!     writes a LeRobot v3.0 dataset

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::datagen::{DemoOptions, Demonstration, Origin, demonstrations};
use batchplan::npy::write_npy;
use batchplan::rng::Rng;
use batchplan::*;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let lerobot = args.first().is_some_and(|a| a == "--lerobot");
    if lerobot {
        args.remove(0);
    }
    let mut args = args.into_iter();
    let out = PathBuf::from(args.next().context("usage: datagen [--lerobot] <out_dir> [episodes] [fps]")?);
    let episodes: usize = args.next().map_or(Ok(256), |a| a.parse())?;
    let fps: u32 = args.next().map_or(Ok(20), |a| a.parse())?;

    let robot = Robot::from_config_file(common::panda_config())?;
    let device = Device::gpu(&robot).unwrap_or_else(|e| {
        eprintln!("no GPU ({e}); using the CPU");
        Device::cpu(&robot)
    });
    println!("device: {}", device.name());
    let mut rng = Rng::new(7);
    let worlds: Vec<World> = (0..episodes).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng) })
        .collect();

    let started = Instant::now();
    let options = DemoOptions { dt: 1.0 / fps as f32, ..Default::default() };
    let demos = demonstrations(&device, &worlds, &goals, &options)?;
    let planning_time = started.elapsed().as_secs_f64();
    let recoveries = demos.iter().filter(|d| matches!(d.origin, Origin::Recovery { .. })).count();
    println!(
        "{episodes} worlds -> {} nominal + {recoveries} recovery demonstrations in {planning_time:.2} s",
        demos.len() - recoveries
    );

    if lerobot {
        export_lerobot(&out, &robot, &worlds, &demos)?;
    } else {
        let provenance = serde_json::json!({"max_acceleration": options.max_acceleration, "device": device.name()});
        write_arrays(&out, &robot, &worlds, &demos, fps, provenance)?;
    }
    println!("wrote {} episodes to {}", demos.len(), out.display());
    Ok(())
}

#[cfg(feature = "lerobot")]
fn export_lerobot(out: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<()> {
    batchplan::lerobot::export(out, robot, worlds, demos, &Default::default())
}

#[cfg(not(feature = "lerobot"))]
fn export_lerobot(_: &Path, _: &Robot, _: &[World], _: &[Demonstration]) -> Result<()> {
    anyhow::bail!("rebuild with `--features lerobot` to write LeRobot datasets")
}

/// Plain arrays: `[episodes, steps, dof]` positions/velocities padded past `length`, plus per-episode columns.
fn write_arrays(
    out: &Path,
    robot: &Robot,
    worlds: &[World],
    demos: &[Demonstration],
    fps: u32,
    provenance: serde_json::Value,
) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let (n, m) = (robot.dof(), demos.len());
    let horizon = demos.iter().map(|d| d.trajectory.len()).max().unwrap_or(0);
    let mut positions = vec![0.0f32; m * horizon * n];
    let mut velocities = vec![0.0f32; m * horizon * n];
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        for h in 0..horizon {
            let (src, dst) = (h.min(t.len() - 1) * n, (i * horizon + h) * n);
            positions[dst..dst + n].copy_from_slice(&t.positions[src..src + n]);
            if h < t.len() {
                velocities[dst..dst + n].copy_from_slice(&t.velocities[src..src + n]);
            }
        }
    }
    let parent = |d: &Demonstration| match d.origin {
        Origin::Nominal => -1,
        Origin::Recovery { parent, .. } => parent as i32,
    };
    let column = |f: &dyn Fn(&Demonstration) -> i32| demos.iter().map(f).collect::<Vec<i32>>();
    write_npy(out.join("positions.npy"), &[m, horizon, n], &positions)?;
    write_npy(out.join("velocities.npy"), &[m, horizon, n], &velocities)?;
    write_npy(out.join("length.npy"), &[m], &column(&|d| d.trajectory.len() as i32))?;
    write_npy(out.join("parent.npy"), &[m], &column(&parent))?;
    write_npy(out.join("world.npy"), &[m], &column(&|d| d.world as i32))?;
    let kinds: Vec<u8> = demos.iter().map(|d| u8::from(parent(d) >= 0)).collect();
    write_npy(out.join("kind.npy"), &[m], &kinds)?;
    let recoveries = kinds.iter().filter(|&&k| k == 1).count();
    let poses: Vec<f32> = demos
        .iter()
        .flat_map(|d| {
            let (p, q) = (d.goal.position, d.goal.rotation);
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        })
        .collect();
    write_npy(out.join("goal_pose.npy"), &[m, 7], &poses)?;
    std::fs::write(out.join("worlds.json"), serde_json::to_string(worlds)?)?;
    let mut meta = serde_json::json!({
        "dt": 1.0 / fps as f32,
        "joint_names": robot.joint_names(),
        "max_velocity": robot.max_velocity(),
        "episodes": m,
        "nominal": m - recoveries,
        "recovery": recoveries,
        "kind": {"0": "nominal", "1": "recovery from a perturbed state of episode `parent`"},
        "goal_pose": "ee frame target: position xyz + quaternion xyzw",
    });
    meta.as_object_mut().unwrap().extend(provenance.as_object().unwrap().clone());
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}
