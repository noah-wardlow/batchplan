//! Demonstrations as plain NumPy arrays: one `.npy` file per field, loadable with `numpy.load`
//! and nothing else. The dependency-free counterpart of [`crate::lerobot`].

use std::io::Write;
use std::path::Path;

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};

use crate::datagen::Demonstration;
use crate::robot::Robot;
use crate::world::World;

#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    /// Extra entries merged into `meta.json`, e.g. generation parameters.
    pub metadata: Map<String, Value>,
}

/// Writes `demos` under `root`, which must not already hold a dataset:
/// - `positions.npy`, `velocities.npy`: `[episodes, steps, dof]` float32, padded past each
///   episode's `length` with its final position and zero velocity.
/// - `length.npy` (int32), `kind.npy` (uint8: 0 nominal, 1 recovery), `parent.npy` (int32: the
///   episode a recovery branches from, -1 otherwise), `world.npy` (int32): `[episodes]`.
/// - `goal_pose.npy`: `[episodes, 7]` float32, target position xyz + quaternion xyzw.
/// - `worlds.json` (the obstacles of every world) and `meta.json`.
///
/// All demonstrations must share one sample period.
pub fn export(root: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration], o: &ExportOptions) -> Result<()> {
    ensure!(!demos.is_empty(), "no demonstrations to export");
    ensure!(!root.join("meta.json").exists(), "{} already holds a dataset", root.display());
    let dt = demos[0].trajectory.dt;
    ensure!(demos.iter().all(|d| d.trajectory.dt == dt), "all demonstrations must share one dt");
    std::fs::create_dir_all(root)?;

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
    let column = |f: &dyn Fn(&Demonstration) -> i32| demos.iter().map(f).collect::<Vec<i32>>();
    write(root.join("positions.npy"), &[m, horizon, n], &positions)?;
    write(root.join("velocities.npy"), &[m, horizon, n], &velocities)?;
    write(root.join("length.npy"), &[m], &column(&|d| d.trajectory.len() as i32))?;
    write(root.join("parent.npy"), &[m], &column(&|d| d.origin.parent().map_or(-1, |p| p as i32)))?;
    write(root.join("world.npy"), &[m], &column(&|d| d.world as i32))?;
    let kinds: Vec<u8> = demos.iter().map(|d| u8::from(d.origin.parent().is_some())).collect();
    write(root.join("kind.npy"), &[m], &kinds)?;
    let poses: Vec<f32> = demos
        .iter()
        .flat_map(|d| {
            let (p, q) = (d.goal.position, d.goal.rotation);
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        })
        .collect();
    write(root.join("goal_pose.npy"), &[m, 7], &poses)?;
    std::fs::write(root.join("worlds.json"), serde_json::to_string(worlds)?)?;

    let recoveries = kinds.iter().filter(|&&k| k == 1).count();
    let mut meta = json!({
        "dt": dt,
        "joint_names": robot.joint_names(),
        "max_velocity": robot.max_velocity(),
        "episodes": m,
        "nominal": m - recoveries,
        "recovery": recoveries,
        "kind": {"0": "nominal", "1": "recovery from a perturbed state of episode `parent`"},
        "goal_pose": "ee frame target: position xyz + quaternion xyzw",
    });
    meta.as_object_mut().expect("meta is an object").extend(o.metadata.clone());
    std::fs::write(root.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

trait Element: bytemuck::Pod {
    const DESCR: &'static str;
}

impl Element for f32 {
    const DESCR: &'static str = "<f4";
}

impl Element for i32 {
    const DESCR: &'static str = "<i4";
}

impl Element for u8 {
    const DESCR: &'static str = "|u1";
}

/// One `.npy` file (format version 1.0, little-endian, C order).
fn write<T: Element>(path: impl AsRef<Path>, shape: &[usize], data: &[T]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == data.len(), "shape {shape:?} does not match {} elements", data.len());
    let dims = match shape {
        [n] => format!("{n},"),
        _ => shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({dims}), }}", T::DESCR);
    // Magic (6) + version (2) + header length (2) + header must be a multiple of 64.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytemuck::cast_slice(data))?;
    f.flush()?;
    Ok(())
}
