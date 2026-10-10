//! Optional bridge to LeRobot: writes [`Demonstration`]s as a LeRobot v3.0 dataset that
//! `lerobot.datasets.LeRobotDataset` loads directly. Enabled by the `lerobot` cargo feature.
//!
//! Each demonstration becomes one episode with one frame per trajectory sample:
//! - `observation.state`: joint positions.
//! - `action`: joint positions at the next frame (the last frame repeats its own).
//! - `observation.environment_state`: the goal pose (position xyz, quaternion xyzw with w >= 0),
//!   then every obstacle of the episode's world as `[present, kind, center xyz, size xyz,
//!   quaternion xyzw]`, zero-padded to the largest world. `kind` is 0 cuboid, 1 sphere, 2 cylinder,
//!   3 capsule, 4 distance grid; `size` is the half extents of a cuboid or of a grid's box, else
//!   (radius, radius, half height or half length).
//! - `is_recovery`, `parent_episode_index` (-1 for nominal episodes) and `world_index`: extra
//!   columns that LeRobot policies ignore.
//!
//! `meta/batchplan.json` additionally holds the worlds (`grids` and `worlds`, as in the `.npy`
//! export's `worlds.json`) and the environment-state layout.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result, ensure_input};
use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::{Map, Value, json};

use crate::datagen::{Demonstration, Origin};
use crate::robot::Robot;
use crate::world::{Obstacle, World, worlds_json};

const CODEBASE_VERSION: &str = "v3.0";
const CHUNKS_SIZE: usize = 1000;
const DATA_FILE_SIZE_IN_MB: usize = 100;
const VIDEO_FILE_SIZE_IN_MB: usize = 200;
const DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
const QUANTILES: [(f64, &str); 5] = [(0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")];
const GOAL_WIDTH: usize = 7;
const OBSTACLE_WIDTH: usize = 12;

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Natural-language task recorded for every episode.
    pub task: String,
    /// `robot_type` in `meta/info.json`; the URDF robot name when `None`.
    pub robot_type: Option<String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { task: "Move the gripper to the target pose.".into(), robot_type: None }
    }
}

/// Writes `demos` as a LeRobot v3.0 dataset under `root`, which must not already hold one.
/// All demonstrations must share a sample period that is a whole number of frames per second.
pub fn export(root: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration], o: &ExportOptions) -> Result<()> {
    ensure_input!(!demos.is_empty(), "no demonstrations to export");
    ensure_input!(!root.join("meta/info.json").exists(), "{} already holds a LeRobot dataset", root.display());
    let dt = demos[0].trajectory.dt;
    let fps = (1.0 / dt).round();
    ensure_input!(
        fps >= 1.0 && (fps * dt - 1.0).abs() < 1e-4,
        "LeRobot needs a whole number of frames per second; dt = {dt} s is {} fps",
        1.0 / dt
    );
    ensure_input!(demos.iter().all(|d| d.trajectory.dt == dt), "all demonstrations must share one dt");
    let fps = fps as u32;
    let max_obstacles = demos.iter().map(|d| worlds[d.world as usize].obstacles.len()).max().unwrap_or(0);

    let mut data = DataFiles::new(root.join("data"));
    let mut episodes = vec![];
    let mut all: Vec<Column> = vec![];
    let mut index = 0usize;
    for (e, demo) in demos.iter().enumerate() {
        let columns = episode_columns(robot, &worlds[demo.world as usize], demo, e, index, fps, max_obstacles);
        let (chunk, file) = data.write(&record_batch(&columns)?)?;
        let length = demo.trajectory.len();
        episodes.push(EpisodeRow {
            length,
            chunk,
            file,
            from: index,
            stats: columns.iter().map(Column::stats).collect(),
        });
        if all.is_empty() {
            all = columns;
        } else {
            all.iter_mut().zip(columns).for_each(|(a, c)| a.append(c));
        }
        index += length;
    }
    data.finish()?;

    let meta = root.join("meta");
    fs::create_dir_all(meta.join("episodes/chunk-000"))?;
    write_parquet(&meta.join("episodes/chunk-000/file-000.parquet"), &episodes_batch(&episodes, &all, &o.task)?)?;
    write_parquet(&meta.join("tasks.parquet"), &tasks_batch(&o.task)?)?;

    let features: Map<String, Value> = all.iter().map(|c| (c.name.clone(), c.feature())).collect();
    let info = json!({
        "codebase_version": CODEBASE_VERSION,
        "robot_type": o.robot_type.clone().unwrap_or_else(|| robot.name().to_string()),
        "total_episodes": demos.len(),
        "total_frames": index,
        "total_tasks": 1,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {"train": format!("0:{}", demos.len())},
        "data_path": DATA_PATH,
        "video_path": null,
        "features": features,
    });
    let stats: Map<String, Value> = all.iter().map(|c| (c.name.clone(), c.stats().to_json())).collect();
    let extension = json!({
        "generator": "batchplan",
        "environment_state": {
            "goal": "[0:7] position xyz, quaternion xyzw (w >= 0)",
            "obstacles": format!(
                "[7 + {OBSTACLE_WIDTH}k : 7 + {OBSTACLE_WIDTH}(k+1)] obstacle k: present, kind (0 cuboid, 1 sphere, 2 cylinder, 3 capsule, 4 distance grid), center xyz, size xyz (half extents of a cuboid or a grid's box, else radius, radius, half height or half length), quaternion xyzw"
            ),
            "max_obstacles": max_obstacles,
        },
        "episodes": demos.iter().map(|d| match d.origin {
            Origin::Nominal => json!({"origin": "nominal"}),
            Origin::Recovery { parent, phase } => json!({"origin": "recovery", "parent": parent, "phase": phase}),
        }).collect::<Vec<_>>(),
        "worlds": worlds_json(worlds)?,
    });
    fs::write(meta.join("info.json"), serde_json::to_string_pretty(&info)?)?;
    fs::write(meta.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    fs::write(meta.join("batchplan.json"), serde_json::to_string(&extension)?)?;
    Ok(())
}

enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// One per-frame feature: `width` values per frame (1 = scalar).
struct Column {
    name: String,
    width: usize,
    names: Option<Vec<String>>,
    data: Data,
}

impl Column {
    fn f32(name: &str, width: usize, names: Option<Vec<String>>, values: Vec<f32>) -> Self {
        Self { name: name.into(), width, names, data: Data::F32(values) }
    }

    fn i64(name: &str, values: Vec<i64>) -> Self {
        Self { name: name.into(), width: 1, names: None, data: Data::I64(values) }
    }

    fn append(&mut self, other: Column) {
        match (&mut self.data, other.data) {
            (Data::F32(a), Data::F32(b)) => a.extend(b),
            (Data::I64(a), Data::I64(b)) => a.extend(b),
            _ => unreachable!("columns are built with a fixed type per feature"),
        }
    }

    fn feature(&self) -> Value {
        let dtype = match self.data {
            Data::F32(_) => "float32",
            Data::I64(_) => "int64",
        };
        json!({"dtype": dtype, "shape": [self.width], "names": self.names})
    }

    fn values(&self) -> Vec<f64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    fn arrow(&self) -> (Field, ArrayRef) {
        match (&self.data, self.width) {
            (Data::F32(v), 1) => {
                (Field::new(&self.name, DataType::Float32, true), Arc::new(Float32Array::from(v.clone())))
            }
            (Data::I64(v), _) => (Field::new(&self.name, DataType::Int64, true), Arc::new(Int64Array::from(v.clone()))),
            (Data::F32(v), w) => {
                let item = Arc::new(Field::new_list_field(DataType::Float32, true));
                let array =
                    FixedSizeListArray::new(item.clone(), w as i32, Arc::new(Float32Array::from(v.clone())), None);
                (Field::new(&self.name, DataType::FixedSizeList(item, w as i32), true), Arc::new(array))
            }
        }
    }

    /// Per-dimension statistics over all frames, as LeRobot computes them.
    fn stats(&self) -> Stats {
        let values = self.values();
        let frames = values.len() / self.width;
        let mut s = Stats { count: frames, ..Default::default() };
        for d in 0..self.width {
            let mut x: Vec<f64> = (0..frames).map(|f| values[f * self.width + d]).collect();
            let mean = x.iter().sum::<f64>() / frames as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / frames as f64;
            x.sort_by(f64::total_cmp);
            s.min.push(x[0]);
            s.max.push(x[frames - 1]);
            s.mean.push(mean);
            s.std.push(var.sqrt());
            for (q, (level, _)) in s.quantiles.iter_mut().zip(QUANTILES) {
                // Linear interpolation between order statistics (numpy's default).
                let pos = level * (frames - 1) as f64;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                q.push(x[lo] + (pos - lo as f64) * (x[hi] - x[lo]));
            }
        }
        s
    }
}

#[derive(Default)]
struct Stats {
    min: Vec<f64>,
    max: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
    count: usize,
    quantiles: [Vec<f64>; 5],
}

impl Stats {
    /// `(name, values)` in LeRobot's order; `count` is reported separately.
    fn vectors(&self) -> Vec<(&'static str, &[f64])> {
        let mut v =
            vec![("min", &self.min[..]), ("max", &self.max[..]), ("mean", &self.mean[..]), ("std", &self.std[..])];
        v.extend(QUANTILES.iter().zip(&self.quantiles).map(|((_, name), q)| (*name, &q[..])));
        v
    }

    fn to_json(&self) -> Value {
        let mut m: Map<String, Value> = self.vectors().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        m.insert("count".into(), json!([self.count]));
        Value::Object(m)
    }
}

fn episode_columns(
    robot: &Robot,
    world: &World,
    demo: &Demonstration,
    episode: usize,
    first_index: usize,
    fps: u32,
    max_obstacles: usize,
) -> Vec<Column> {
    let n = robot.dof();
    let traj = &demo.trajectory;
    let frames = traj.len();
    let state = traj.positions.clone();
    let mut action = traj.positions[n..].to_vec();
    action.extend_from_slice(&traj.positions[(frames - 1) * n..]);
    let env = environment_state(world, demo, max_obstacles);
    let env_width = env.len();
    let (is_recovery, parent) =
        (i64::from(demo.origin.parent().is_some()), demo.origin.parent().map_or(-1, |p| p as i64));
    let joint_names = Some(robot.joint_names().to_vec());
    let per_frame = |v: i64| vec![v; frames];
    vec![
        Column::f32("observation.state", n, joint_names.clone(), state),
        Column::f32(
            "observation.environment_state",
            env_width,
            Some(environment_names(max_obstacles)),
            env.repeat(frames),
        ),
        Column::f32("action", n, joint_names, action),
        Column::i64("is_recovery", per_frame(is_recovery)),
        Column::i64("parent_episode_index", per_frame(parent)),
        Column::i64("world_index", per_frame(demo.world as i64)),
        Column::f32("timestamp", 1, None, (0..frames).map(|f| f as f32 / fps as f32).collect()),
        Column::i64("frame_index", (0..frames as i64).collect()),
        Column::i64("episode_index", per_frame(episode as i64)),
        Column::i64("index", (first_index as i64..(first_index + frames) as i64).collect()),
        Column::i64("task_index", per_frame(0)),
    ]
}

fn environment_state(world: &World, demo: &Demonstration, max_obstacles: usize) -> Vec<f32> {
    let canonical = |q: glam::Quat| if q.w < 0.0 { -q } else { q };
    let (p, q) = (demo.goal.position, canonical(demo.goal.rotation));
    let mut env = vec![p.x, p.y, p.z, q.x, q.y, q.z, q.w];
    for o in &world.obstacles {
        let round = |radius: f32, half: f32| glam::Vec3::new(radius, radius, half);
        let (kind, center, size, rot) = match *o {
            Obstacle::Cuboid { center, half_extents, rotation } => (0.0, center, half_extents, rotation),
            Obstacle::Sphere { center, radius } => (1.0, center, glam::Vec3::splat(radius), glam::Quat::IDENTITY),
            Obstacle::Cylinder { center, rotation, radius, half_height } => {
                (2.0, center, round(radius, half_height), rotation)
            }
            Obstacle::Capsule { center, rotation, radius, half_length } => {
                (3.0, center, round(radius, half_length), rotation)
            }
            Obstacle::Sdf { ref grid, center, rotation } => {
                let (lo, hi) = grid.bounds();
                (4.0, center + rotation * (0.5 * (lo + hi)), 0.5 * (hi - lo), rotation)
            }
        };
        let rot = canonical(rot);
        env.extend([1.0, kind, center.x, center.y, center.z, size.x, size.y, size.z, rot.x, rot.y, rot.z, rot.w]);
    }
    env.resize(GOAL_WIDTH + OBSTACLE_WIDTH * max_obstacles, 0.0);
    env
}

fn environment_names(max_obstacles: usize) -> Vec<String> {
    let mut names: Vec<String> = ["x", "y", "z", "qx", "qy", "qz", "qw"].iter().map(|s| format!("goal.{s}")).collect();
    for k in 0..max_obstacles {
        let fields = ["present", "kind", "x", "y", "z", "sx", "sy", "sz", "qx", "qy", "qz", "qw"];
        names.extend(fields.iter().map(|f| format!("obstacle{k}.{f}")));
    }
    names
}

fn record_batch(columns: &[Column]) -> Result<RecordBatch> {
    let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = columns.iter().map(Column::arrow).unzip();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}

struct EpisodeRow {
    length: usize,
    chunk: usize,
    file: usize,
    from: usize,
    stats: Vec<Stats>,
}

/// `meta/episodes` rows: boundaries into the data files plus flattened per-episode stats.
fn episodes_batch(episodes: &[EpisodeRow], features: &[Column], task: &str) -> Result<RecordBatch> {
    let int_column = |f: &dyn Fn(usize, &EpisodeRow) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from(episodes.iter().enumerate().map(|(i, e)| f(i, e) as i64).collect::<Vec<_>>()))
    };
    let mut tasks = ListBuilder::new(StringBuilder::new());
    for _ in episodes {
        tasks.values().append_value(task);
        tasks.append(true);
    }
    let mut columns: Vec<(String, ArrayRef)> = vec![
        ("episode_index".into(), int_column(&|i, _| i)),
        ("tasks".into(), Arc::new(tasks.finish())),
        ("length".into(), int_column(&|_, e| e.length)),
        ("data/chunk_index".into(), int_column(&|_, e| e.chunk)),
        ("data/file_index".into(), int_column(&|_, e| e.file)),
        ("dataset_from_index".into(), int_column(&|_, e| e.from)),
        ("dataset_to_index".into(), int_column(&|_, e| e.from + e.length)),
        ("meta/episodes/chunk_index".into(), int_column(&|_, _| 0)),
        ("meta/episodes/file_index".into(), int_column(&|_, _| 0)),
    ];
    for (f, feature) in features.iter().enumerate() {
        let names: Vec<&str> = episodes[0].stats[f].vectors().iter().map(|(n, _)| *n).collect();
        for (s, stat) in names.iter().enumerate() {
            let mut b = ListBuilder::new(Float64Builder::new());
            for e in episodes {
                b.values().append_slice(e.stats[f].vectors()[s].1);
                b.append(true);
            }
            columns.push((format!("stats/{}/{stat}", feature.name), Arc::new(b.finish())));
        }
        let mut count = ListBuilder::new(Int64Builder::new());
        for e in episodes {
            count.values().append_value(e.stats[f].count as i64);
            count.append(true);
        }
        columns.push((format!("stats/{}/count", feature.name), Arc::new(count.finish())));
    }
    let fields: Vec<Field> = columns.iter().map(|(n, a)| Field::new(n, a.data_type().clone(), true)).collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, a)| a).collect())?)
}

/// `meta/tasks.parquet` as pandas writes it: a `task_index` column indexed by the task string.
fn tasks_batch(task: &str) -> Result<RecordBatch> {
    let pandas = json!({
        "index_columns": ["task"],
        "column_indexes": [{"name": null, "field_name": null, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": null},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": null},
        ],
        "creator": {"library": "batchplan", "version": env!("CARGO_PKG_VERSION")},
        "pandas_version": "2.2.3",
    });
    let schema =
        Schema::new(vec![Field::new("task_index", DataType::Int64, true), Field::new("task", DataType::Utf8, true)])
            .with_metadata(HashMap::from([("pandas".to_string(), pandas.to_string())]));
    let arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![0])), Arc::new(StringArray::from(vec![task]))];
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

fn properties() -> WriterProperties {
    WriterProperties::builder().set_compression(Compression::SNAPPY).build()
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    let mut w = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(properties()))?;
    w.write(batch)?;
    w.close()?;
    Ok(())
}

/// Frame data split across `data/chunk-XXX/file-YYY.parquet`, one row group per episode, starting
/// a new file once the current one would pass the size limit.
struct DataFiles {
    dir: PathBuf,
    chunk: usize,
    file: usize,
    bytes: usize,
    writer: Option<ArrowWriter<File>>,
}

impl DataFiles {
    fn new(dir: PathBuf) -> Self {
        Self { dir, chunk: 0, file: 0, bytes: 0, writer: None }
    }

    /// Appends one episode; returns the (chunk, file) it landed in.
    fn write(&mut self, batch: &RecordBatch) -> Result<(usize, usize)> {
        let size = batch.get_array_memory_size();
        if self.writer.is_some() && self.bytes + size > DATA_FILE_SIZE_IN_MB << 20 {
            self.finish()?;
            self.file += 1;
            if self.file == CHUNKS_SIZE {
                (self.chunk, self.file) = (self.chunk + 1, 0);
            }
        }
        if self.writer.is_none() {
            let dir = self.dir.join(format!("chunk-{:03}", self.chunk));
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("file-{:03}.parquet", self.file));
            let file = File::create(&path).map_err(|e| Error::Write(format!("creating {}: {e}", path.display())))?;
            self.writer = Some(ArrowWriter::try_new(file, batch.schema(), Some(properties()))?);
            self.bytes = 0;
        }
        let w = self.writer.as_mut().expect("writer opened above");
        w.write(batch)?;
        w.flush()?;
        self.bytes += size;
        Ok((self.chunk, self.file))
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        Ok(())
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(e: parquet::errors::ParquetError) -> Self {
        Error::Write(e.to_string())
    }
}

impl From<arrow_schema::ArrowError> for Error {
    fn from(e: arrow_schema::ArrowError) -> Self {
        Error::Write(e.to_string())
    }
}
