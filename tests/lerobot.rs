//! LeRobot export (requires `--features lerobot`). `scripts/validate_lerobot.py` checks the same
//! output with the real `lerobot` package.
#![cfg(feature = "lerobot")]

#[path = "../examples/common/mod.rs"]
mod common;

use std::fs::File;

use batchplan::datagen::{DemoOptions, Origin, demonstrations};
use batchplan::lerobot::{ExportOptions, export};
use batchplan::rng::Rng;
use batchplan::*;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn demos(dt: f32) -> (Robot, Vec<World>, Vec<batchplan::datagen::Demonstration>) {
    let robot = Robot::from_config_file(common::panda_config()).unwrap();
    let device = Device::cpu(&robot);
    let mut rng = Rng::new(3);
    let worlds: Vec<World> = (0..6).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng) })
        .collect();
    let o = DemoOptions { dt, ..Default::default() };
    let demos = demonstrations(&device, &worlds, &goals, &o).unwrap();
    (robot, worlds, demos)
}

#[test]
fn export_writes_a_consistent_v3_dataset() {
    let (robot, worlds, demos) = demos(0.05);
    assert!(demos.iter().any(|d| matches!(d.origin, Origin::Recovery { .. })));
    let root = std::env::temp_dir().join(format!("batchplan-lerobot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    export(&root, &robot, &worlds, &demos, &ExportOptions::default()).unwrap();

    let info: serde_json::Value = serde_json::from_reader(File::open(root.join("meta/info.json")).unwrap()).unwrap();
    let frames: usize = demos.iter().map(|d| d.trajectory.len()).sum();
    assert_eq!(info["codebase_version"], "v3.0");
    assert_eq!(info["fps"], 20);
    assert_eq!(info["total_episodes"], demos.len());
    assert_eq!(info["total_frames"], frames);
    assert_eq!(info["features"]["observation.state"]["shape"][0], robot.dof());

    let rows = |path: &str| {
        SerializedFileReader::new(File::open(root.join(path)).unwrap()).unwrap().metadata().file_metadata().num_rows()
    };
    assert_eq!(rows("data/chunk-000/file-000.parquet") as usize, frames);
    assert_eq!(rows("meta/episodes/chunk-000/file-000.parquet") as usize, demos.len());
    assert_eq!(rows("meta/tasks.parquet"), 1);
    let data = SerializedFileReader::new(File::open(root.join("data/chunk-000/file-000.parquet")).unwrap()).unwrap();
    assert_eq!(data.metadata().num_row_groups(), demos.len(), "one row group per episode");

    // Exporting over an existing dataset is refused.
    assert!(export(&root, &robot, &worlds, &demos, &ExportOptions::default()).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn export_rejects_fractional_frame_rates() {
    let (robot, worlds, demos) = demos(0.03);
    let root = std::env::temp_dir().join(format!("batchplan-lerobot-fps-{}", std::process::id()));
    let err = export(&root, &robot, &worlds, &demos, &ExportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("frames per second"), "{err}");
}
