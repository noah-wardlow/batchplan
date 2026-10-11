//! Generates policy-training demonstrations: reach-to-grasp trajectories in random cluttered
//! worlds plus recoveries from perturbed mid-path states, timed within the robot's limits at
//! randomized speed; or, with `--pick-place`, boxes picked up and set down across a table.
//!
//! cargo run --release --example datagen -- [--pick-place] <out_dir> [episodes=256] [fps=20]
//!     writes plain NumPy `.npy` arrays
//! cargo run --release --features lerobot --example datagen -- --lerobot [--file-mb N] [--images [--depth]] <out_dir> [episodes] [fps]
//!     writes a LeRobot v3.0 dataset, starting new files past N MB (default 100), with 128x128
//!     colour images from a front camera and a wrist camera with `--images`, and depth images too
//!     with `--depth`

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::datagen::{DemoOptions, Demonstration, Origin, PickPlaceOptions, demonstrations, pick_and_place};
use batchplan::rng::Rng;
use batchplan::*;

fn main() -> Result<()> {
    let (mut lerobot, mut pick_place, mut images, mut depth, mut file_mb, mut positional) =
        (false, false, false, false, 100, vec![]);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lerobot" => lerobot = true,
            "--pick-place" => pick_place = true,
            "--images" => images = true,
            "--depth" => depth = true,
            "--file-mb" => file_mb = args.next().context("--file-mb N")?.parse()?,
            _ => positional.push(arg),
        }
    }
    let mut args = positional.into_iter();
    let out = PathBuf::from(
        args.next().context("usage: datagen [--pick-place] [--lerobot [--file-mb N]] <out_dir> [episodes] [fps]")?,
    );
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
    let started = Instant::now();
    let (worlds, demos) = if pick_place {
        let (worlds, tasks): (Vec<World>, Vec<_>) =
            (0..episodes).map(|i| common::pick_place_scene(i as u32, &mut rng)).unzip();
        let options = PickPlaceOptions { dt: 1.0 / fps as f32, ..Default::default() };
        let demos = pick_and_place(&device, &worlds, &tasks, &options)?;
        println!(
            "{episodes} tasks -> {} pick-and-place demonstrations in {:.2} s",
            demos.len(),
            started.elapsed().as_secs_f64()
        );
        (worlds, demos)
    } else {
        let worlds: Vec<World> = (0..episodes).map(|_| common::tabletop(&mut rng)).collect();
        let goals: Vec<IkProblem> = worlds
            .iter()
            .enumerate()
            .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng), seed: None })
            .collect();
        let options = DemoOptions { dt: 1.0 / fps as f32, ..Default::default() };
        let demos = demonstrations(&device, &device.upload(&worlds)?, &goals, &options)?;
        let recoveries = demos.iter().filter(|d| matches!(d.origin, Origin::Recovery { .. })).count();
        println!(
            "{episodes} worlds -> {} nominal + {recoveries} recovery demonstrations in {:.2} s",
            demos.len() - recoveries,
            started.elapsed().as_secs_f64()
        );
        (worlds, demos)
    };

    if lerobot {
        export_lerobot(&out, &device, &worlds, &demos, file_mb, images, depth)?;
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
fn export_lerobot(
    out: &Path,
    device: &Device,
    worlds: &[World],
    demos: &[Demonstration],
    file_mb: usize,
    images: bool,
    depth_images: bool,
) -> Result<()> {
    let intrinsics = Intrinsics { fx: 110.0, fy: 110.0, cx: 63.5, cy: 63.5 };
    let eye = glam::Vec3::new(1.3, 0.0, 0.7);
    let forward = (glam::Vec3::new(0.4, 0.0, 0.1) - eye).normalize();
    let right = forward.cross(glam::Vec3::Z).normalize();
    let front = Pose {
        position: eye,
        rotation: glam::Quat::from_mat3(&glam::Mat3::from_cols(right, forward.cross(right), forward)),
    };
    // Beside the hand, tilted toward the grasp point between the fingertips.
    let wrist = Pose { position: glam::Vec3::new(0.06, 0.0, 0.0), rotation: glam::Quat::from_rotation_y(-0.54) };
    let camera = |name: &str, mount| Camera { name: name.into(), width: 128, height: 128, intrinsics, mount };
    let cameras = if images {
        vec![
            camera("front", Mount::World(front)),
            camera("wrist", Mount::Link { link: "panda_hand".into(), offset: wrist }),
        ]
    } else {
        vec![]
    };
    let options = batchplan::lerobot::ExportOptions {
        data_files_size_in_mb: file_mb,
        cameras,
        depth_images,
        ..Default::default()
    };
    Ok(batchplan::lerobot::export(out, device, &device.upload(worlds)?, demos, &options)?)
}

#[cfg(not(feature = "lerobot"))]
fn export_lerobot(_: &Path, _: &Device, _: &[World], _: &[Demonstration], _: usize, _: bool, _: bool) -> Result<()> {
    anyhow::bail!("rebuild with `--features lerobot` to write LeRobot datasets")
}
