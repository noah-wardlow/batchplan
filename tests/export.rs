//! Dataset exporters. The LeRobot half needs `--features lerobot`; `scripts/validate_lerobot.py`
//! additionally checks that output with the real `lerobot` package.

#[path = "../examples/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};

use batchplan::datagen::{DemoOptions, Demonstration, demonstrations};
use batchplan::rng::Rng;
use batchplan::*;

fn demos(dt: f32) -> (Robot, Vec<World>, Vec<Demonstration>) {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let mut rng = Rng::new(3);
    let worlds: Vec<World> = (0..6).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng) })
        .collect();
    let uploaded = device.upload(&worlds).unwrap();
    let demos = demonstrations(&device, &uploaded, &goals, &DemoOptions { dt, ..Default::default() }).unwrap();
    assert!(demos.iter().any(|d| d.origin.parent().is_some()), "fixture should include recoveries");
    (robot, worlds, demos)
}

fn scratch(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("batchplan-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    root
}

/// Reads a `.npy` file written by the exporter: (shape, raw little-endian data).
fn read_npy(path: &Path) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0, "header must be 64-byte aligned");
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let shape = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape = shape.split(',').filter_map(|d| d.trim().parse().ok()).collect();
    (shape, bytes[10 + header_len..].to_vec())
}

fn as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn as_i32(data: &[u8]) -> Vec<i32> {
    data.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn npy_export_round_trips_the_demonstrations() {
    let (robot, worlds, demos) = demos(0.05);
    let root = scratch("npy");
    batchplan::npy::export(&root, &robot, &worlds, &demos, &Default::default()).unwrap();

    let n = robot.dof();
    let horizon = demos.iter().map(|d| d.trajectory.len()).max().unwrap();
    let (shape, positions) = read_npy(&root.join("positions.npy"));
    assert_eq!(shape, [demos.len(), horizon, n]);
    let positions = as_f32(&positions);
    let lengths = as_i32(&read_npy(&root.join("length.npy")).1);
    let parents = as_i32(&read_npy(&root.join("parent.npy")).1);
    let kinds = read_npy(&root.join("kind.npy")).1;
    let worlds_column = as_i32(&read_npy(&root.join("world.npy")).1);
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        assert_eq!(lengths[i] as usize, t.len());
        assert_eq!(parents[i], d.origin.parent().map_or(-1, |p| p as i32));
        assert_eq!(kinds[i], u8::from(d.origin.parent().is_some()));
        assert_eq!(worlds_column[i], d.world as i32);
        let row = &positions[i * horizon * n..(i + 1) * horizon * n];
        assert_eq!(&row[..t.positions.len()], &t.positions[..], "episode {i} positions");
        // Padding repeats the final position.
        assert!(row[t.positions.len()..].chunks(n).all(|q| q == &t.positions[t.positions.len() - n..]));
    }
    // Exporting over an existing dataset is refused.
    assert!(batchplan::npy::export(&root, &robot, &worlds, &demos, &Default::default()).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

#[cfg(feature = "lerobot")]
mod lerobot {
    use std::fs::File;

    use batchplan::lerobot::{ExportOptions, export};
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;

    #[test]
    fn export_writes_a_consistent_v3_dataset() {
        let (robot, worlds, demos) = demos(0.05);
        let root = scratch("lerobot");
        export(&root, &robot, &worlds, &demos, &ExportOptions::default()).unwrap();

        let info: serde_json::Value =
            serde_json::from_reader(File::open(root.join("meta/info.json")).unwrap()).unwrap();
        let frames: usize = demos.iter().map(|d| d.trajectory.len()).sum();
        assert_eq!(info["codebase_version"], "v3.0");
        assert_eq!(info["fps"], 20);
        assert_eq!(info["total_episodes"], demos.len());
        assert_eq!(info["total_frames"], frames);
        assert_eq!(info["features"]["observation.state"]["shape"][0], robot.dof());

        let reader = |path: &str| SerializedFileReader::new(File::open(root.join(path)).unwrap()).unwrap();
        let rows = |path: &str| reader(path).metadata().file_metadata().num_rows() as usize;
        assert_eq!(rows("data/chunk-000/file-000.parquet"), frames);
        assert_eq!(rows("meta/episodes/chunk-000/file-000.parquet"), demos.len());
        assert_eq!(rows("meta/tasks.parquet"), 1);
        let data = reader("data/chunk-000/file-000.parquet");
        assert_eq!(data.metadata().num_row_groups(), demos.len(), "one row group per episode");

        // Exporting over an existing dataset is refused.
        assert!(export(&root, &robot, &worlds, &demos, &ExportOptions::default()).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn export_rejects_fractional_frame_rates() {
        let (robot, worlds, demos) = demos(0.03);
        let err = export(&scratch("lerobot-fps"), &robot, &worlds, &demos, &ExportOptions::default()).unwrap_err();
        assert!(err.to_string().contains("frames per second"), "{err}");
    }
}
