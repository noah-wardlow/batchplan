//! Generates policy-training demonstrations: reach-to-grasp trajectories in random cluttered
//! worlds plus recoveries from perturbed mid-path states, timed within the robot's limits at
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

    let robot = common::panda()?;
    let device = match Device::gpu(&robot) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}); using the CPU");
            Device::cpu(&robot)?
        }
    };
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
    let demos = demonstrations(&device, &device.upload(&worlds)?, &goals, &options)?;
    let planning_time = started.elapsed().as_secs_f64();
    let recoveries = demos.iter().filter(|d| matches!(d.origin, Origin::Recovery { .. })).count();
    println!(
        "{episodes} worlds -> {} nominal + {recoveries} recovery demonstrations in {planning_time:.2} s",
        demos.len() - recoveries
    );

    if lerobot {
        export_lerobot(&out, &robot, &worlds, &demos)?;
    } else {
        let metadata = serde_json::json!({
            "max_acceleration": robot.max_acceleration(),
            "max_jerk": robot.max_jerk(),
            "device": device.name(),
        });
        let metadata = metadata.as_object().expect("an object").clone();
        batchplan::npy::export(&out, &robot, &worlds, &demos, &batchplan::npy::ExportOptions { metadata })?;
    }
    println!("wrote {} episodes to {}", demos.len(), out.display());
    Ok(())
}

#[cfg(feature = "lerobot")]
fn export_lerobot(out: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<()> {
    Ok(batchplan::lerobot::export(out, robot, worlds, demos, &Default::default())?)
}

#[cfg(not(feature = "lerobot"))]
fn export_lerobot(_: &Path, _: &Robot, _: &[World], _: &[Demonstration]) -> Result<()> {
    anyhow::bail!("rebuild with `--features lerobot` to write LeRobot datasets")
}
