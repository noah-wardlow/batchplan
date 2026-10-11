//! Optional bridge to LeRobot: writes [`Demonstration`]s as a LeRobot v3.0 dataset that
//! `lerobot.datasets.LeRobotDataset` loads directly. Enabled by the `lerobot` cargo feature.
//!
//! Each demonstration becomes one episode with one frame per trajectory sample, and its task one
//! of the dataset's tasks:
//! - `observation.state`: joint positions, then the gripper's opening (1 open, 0 closed).
//! - `action`: the state at the next frame (the last frame repeats its own).
//! - `observation.environment_state`: the goal pose (position xyz, quaternion xyzw with w >= 0),
//!   then every obstacle of the episode's world, where it is at that frame, as `[present, kind,
//!   center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world. `kind` is 0 cuboid,
//!   1 sphere, 2 cylinder, 3 capsule, 4 distance grid; `size` is the half extents of a cuboid or of
//!   a grid's box, else (radius, radius, half height or half length).
//! - `observation.images.<camera>` for each of [`ExportOptions::cameras`]: the colour image
//!   `[height, width, 3]` as a PNG `image` feature, rendered on the device; with
//!   [`ExportOptions::depth_images`], also `observation.images.<camera>_depth`, the depth along
//!   the camera's axis in millimetres `[height, width, 1]` (16-bit, flagged `is_depth_map`, 0
//!   where nothing is hit).
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
use arrow_array::{
    ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use serde_json::{Map, Value, json};

use crate::datagen::{Demonstration, Origin, check_demos};
use crate::device::{Device, Worlds};
use crate::render::{Camera, Images};
use crate::robot::Robot;
use crate::types::Pose;
use crate::world::{Obstacle, World, worlds_json};

const CODEBASE_VERSION: &str = "v3.0";
const CHUNKS_SIZE: usize = 1000;
const VIDEO_FILE_SIZE_IN_MB: usize = 200;
/// Episode-metadata rows written at a time.
const EPISODES_PER_WRITE: usize = 1000;
const DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
const QUANTILES: [(f64, &str); 5] = [(0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")];
const GOAL_WIDTH: usize = 7;
const OBSTACLE_WIDTH: usize = 12;

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// `robot_type` in `meta/info.json`; the URDF robot name when `None`.
    pub robot_type: Option<String>,
    /// Frame data and episode metadata start a new file past this size, as LeRobot's own
    /// `data_files_size_in_mb`.
    pub data_files_size_in_mb: usize,
    /// Cameras whose images every frame records; none by default.
    pub cameras: Vec<Camera>,
    /// Whether each camera also records depth images. LeRobot 0.6's stock image policies (ACT,
    /// diffusion) read every camera feature as colour, so this is off by default.
    pub depth_images: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { robot_type: None, data_files_size_in_mb: 100, cameras: vec![], depth_images: false }
    }
}

/// Writes `demos` of `device`'s robot in `worlds` (uploaded to `device`, which renders the
/// cameras' images) as a LeRobot v3.0 dataset under `root`, which must not already hold one. All
/// demonstrations must share a sample period that is a whole number of frames per second. Frames
/// and episodes are written as they are converted, so the dataset can exceed memory.
pub fn export(root: &Path, device: &Device, worlds: &Worlds, demos: &[Demonstration], o: &ExportOptions) -> Result<()> {
    let robot = device.robot();
    let uploaded = worlds;
    let worlds = worlds.as_slice();
    let dt = check_demos(robot, worlds, demos)?;
    for (i, camera) in o.cameras.iter().enumerate() {
        camera.check(robot)?;
        ensure_input!(o.cameras[..i].iter().all(|c| c.name != camera.name), "two cameras are named '{}'", camera.name);
    }
    ensure_input!(!root.join("meta/info.json").exists(), "{} already holds a LeRobot dataset", root.display());
    ensure_input!(o.data_files_size_in_mb > 0, "data files need a positive size limit");
    let fps = (1.0 / dt).round();
    ensure_input!(
        fps >= 1.0 && (fps * dt - 1.0).abs() < 1e-4,
        "LeRobot needs a whole number of frames per second; dt = {dt} s is {} fps",
        1.0 / dt
    );
    let fps = fps as u32;
    let max_obstacles = demos.iter().map(|d| worlds[d.world as usize].obstacles.len()).max().unwrap_or(0);
    // Tasks in order of first appearance.
    let mut tasks: Vec<&str> = vec![];
    let mut task_index: HashMap<&str, usize> = HashMap::new();
    for d in demos {
        task_index.entry(&d.task).or_insert_with(|| {
            tasks.push(&d.task);
            tasks.len() - 1
        });
    }

    let limit = o.data_files_size_in_mb << 20;
    let mut data = Files::new(root.join("data"), limit);
    let mut episode_files = Files::new(root.join("meta/episodes"), limit);
    // Features in column order (`Map` sorts its keys) and their dataset statistics.
    let (mut rows, mut names, mut features, mut totals) = (vec![], vec![], Map::new(), vec![]);
    let mut index = 0usize;
    for (e, demo) in demos.iter().enumerate() {
        let world = &worlds[demo.world as usize];
        let mut columns =
            episode_columns(robot, world, demo, e, index, fps, max_obstacles, task_index[demo.task.as_str()]);
        for camera in &o.cameras {
            let [colour, depth] = image_columns(device, uploaded, camera, demo)?;
            columns.push(colour);
            columns.extend(o.depth_images.then_some(depth));
        }
        let batch = record_batch(&columns)?;
        let (chunk, file) = data.place(batch.get_array_memory_size())?;
        data.write(&batch)?;
        let stats: Vec<Stats> = columns.iter().map(Column::stats).collect();
        if e == 0 {
            names = columns.iter().map(|c| c.name.clone()).collect();
            features = columns.iter().map(|c| (c.name.clone(), c.feature())).collect();
            totals = stats.iter().map(|_| Aggregate::default()).collect();
        }
        totals.iter_mut().zip(&stats).for_each(|(t, s)| t.add(s));
        let length = demo.trajectory.len();
        rows.push(EpisodeRow { length, chunk, file, from: index, task: demo.task.clone(), stats });
        if rows.len() == EPISODES_PER_WRITE {
            write_episodes(&mut episode_files, e + 1 - rows.len(), &rows, &names)?;
            rows.clear();
        }
        index += length;
    }
    if !rows.is_empty() {
        write_episodes(&mut episode_files, demos.len() - rows.len(), &rows, &names)?;
    }
    data.finish()?;
    episode_files.finish()?;

    let meta = root.join("meta");
    write_parquet(&meta.join("tasks.parquet"), &tasks_batch(&tasks)?)?;
    let info = json!({
        "codebase_version": CODEBASE_VERSION,
        "robot_type": o.robot_type.clone().unwrap_or_else(|| robot.name().to_string()),
        "total_episodes": demos.len(),
        "total_frames": index,
        "total_tasks": tasks.len(),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": o.data_files_size_in_mb,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {"train": format!("0:{}", demos.len())},
        "data_path": DATA_PATH,
        "video_path": null,
        "features": features,
    });
    let stats: Map<String, Value> =
        names.iter().zip(&totals).map(|(name, t)| (name.clone(), t.stats().to_json())).collect();
    let extension = json!({
        "generator": "batchplan",
        "environment_state": {
            "goal": "[0:7] position xyz, quaternion xyzw (w >= 0)",
            "obstacles": format!(
                "[7 + {OBSTACLE_WIDTH}k : 7 + {OBSTACLE_WIDTH}(k+1)] obstacle k where it is at the frame: present, kind (0 cuboid, 1 sphere, 2 cylinder, 3 capsule, 4 distance grid), center xyz, size xyz (half extents of a cuboid or a grid's box, else radius, radius, half height or half length), quaternion xyzw"
            ),
            "max_obstacles": max_obstacles,
        },
        "episodes": demos.iter().map(|d| {
            let mut e = match d.origin {
                Origin::Nominal => json!({"origin": "nominal"}),
                Origin::Recovery { parent, phase } => json!({"origin": "recovery", "parent": parent, "phase": phase}),
            };
            if let Some(c) = &d.carried {
                e["carried_obstacle"] = json!(c.obstacle);
            }
            e
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
    /// One PNG per frame, with the pixels' statistics per channel.
    Image {
        png: Vec<Vec<u8>>,
        height: u32,
        depth_map: bool,
        stats: Box<Stats>,
    },
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

    fn feature(&self) -> Value {
        let dtype = match self.data {
            Data::F32(_) => "float32",
            Data::I64(_) => "int64",
            Data::Image { height, depth_map, .. } => {
                let info = depth_map.then(|| json!({"is_depth_map": true, "depth_unit": "mm"}));
                let channels = if depth_map { 1 } else { 3 };
                return json!({
                    "dtype": "image",
                    "shape": [height, self.width, channels],
                    "names": ["height", "width", "channel"],
                    "info": info,
                });
            }
        };
        json!({"dtype": dtype, "shape": [self.width], "names": self.names})
    }

    fn values(&self) -> Vec<f64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Data::Image { .. } => unreachable!("image statistics come from their histograms"),
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
            // The Hugging Face `Image` feature: PNG bytes and no path.
            (Data::Image { png, .. }, _) => {
                let fields = Fields::from(vec![
                    Field::new("bytes", DataType::Binary, true),
                    Field::new("path", DataType::Utf8, true),
                ]);
                let bytes: ArrayRef = Arc::new(BinaryArray::from_iter_values(png.iter()));
                let paths: ArrayRef = Arc::new(StringArray::new_null(png.len()));
                let array = StructArray::new(fields.clone(), vec![bytes, paths], None);
                (Field::new(&self.name, DataType::Struct(fields), true), Arc::new(array))
            }
        }
    }

    /// Per-dimension statistics over all frames, as LeRobot computes them.
    fn stats(&self) -> Stats {
        if let Data::Image { stats, .. } = &self.data {
            return stats.as_ref().clone();
        }
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

/// Dataset statistics from per-episode ones, combined as LeRobot's `aggregate_feature_stats`
/// combines them: count-weighted means and quantiles, and the variance of the union.
#[derive(Default)]
struct Aggregate {
    count: f64,
    image: bool,
    min: Vec<f64>,
    max: Vec<f64>,
    /// Count-weighted sums of the means, of the second moments and of each quantile.
    mean: Vec<f64>,
    second: Vec<f64>,
    quantiles: [Vec<f64>; 5],
}

impl Aggregate {
    fn add(&mut self, s: &Stats) {
        let c = s.count as f64;
        if self.min.is_empty() {
            let zeros = vec![0.0; s.mean.len()];
            (self.min, self.max) = (s.min.clone(), s.max.clone());
            (self.mean, self.second) = (zeros.clone(), zeros.clone());
            self.quantiles = std::array::from_fn(|_| zeros.clone());
        }
        self.count += c;
        self.image = s.image;
        for d in 0..s.mean.len() {
            self.min[d] = self.min[d].min(s.min[d]);
            self.max[d] = self.max[d].max(s.max[d]);
            self.mean[d] += c * s.mean[d];
            self.second[d] += c * (s.std[d].powi(2) + s.mean[d].powi(2));
            for (q, sq) in self.quantiles.iter_mut().zip(&s.quantiles) {
                q[d] += c * sq[d];
            }
        }
    }

    fn stats(&self) -> Stats {
        let mean: Vec<f64> = self.mean.iter().map(|m| m / self.count).collect();
        let std = self.second.iter().zip(&mean).map(|(s, m)| (s / self.count - m * m).max(0.0).sqrt()).collect();
        Stats {
            min: self.min.clone(),
            max: self.max.clone(),
            mean,
            std,
            count: self.count as usize,
            quantiles: self.quantiles.clone().map(|q| q.iter().map(|v| v / self.count).collect()),
            image: self.image,
        }
    }
}

#[derive(Clone, Default)]
struct Stats {
    min: Vec<f64>,
    max: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
    count: usize,
    quantiles: [Vec<f64>; 5],
    /// An image's statistics, one per channel, which LeRobot nests as `[channels][1][1]`.
    image: bool,
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
        let shape = |v: &[f64]| if self.image { json!(v.iter().map(|x| [[x]]).collect::<Vec<_>>()) } else { json!(v) };
        let mut m: Map<String, Value> = self.vectors().into_iter().map(|(k, v)| (k.to_string(), shape(v))).collect();
        m.insert("count".into(), json!([self.count]));
        Value::Object(m)
    }
}

#[allow(clippy::too_many_arguments)]
fn episode_columns(
    robot: &Robot,
    world: &World,
    demo: &Demonstration,
    episode: usize,
    first_index: usize,
    fps: u32,
    max_obstacles: usize,
    task: usize,
) -> Vec<Column> {
    let n = robot.dof();
    let traj = &demo.trajectory;
    let frames = traj.len();
    let state: Vec<f32> =
        (0..frames).flat_map(|f| traj.positions[f * n..(f + 1) * n].iter().copied().chain([demo.gripper[f]])).collect();
    let mut action = state[n + 1..].to_vec();
    action.extend_from_slice(&state[(frames - 1) * (n + 1)..]);
    let env: Vec<f32> = (0..frames)
        .flat_map(|f| {
            let carried = demo.carried.as_ref().map(|c| (c.obstacle, c.poses[f]));
            environment_state(world, demo.goal, carried, max_obstacles)
        })
        .collect();
    let (is_recovery, parent) =
        (i64::from(demo.origin.parent().is_some()), demo.origin.parent().map_or(-1, |p| p as i64));
    let mut names = robot.joint_names().to_vec();
    names.push("gripper".into());
    let per_frame = |v: i64| vec![v; frames];
    vec![
        Column::f32("observation.state", n + 1, Some(names.clone()), state),
        Column::f32("observation.environment_state", env.len() / frames, Some(environment_names(max_obstacles)), env),
        Column::f32("action", n + 1, Some(names), action),
        Column::i64("is_recovery", per_frame(is_recovery)),
        Column::i64("parent_episode_index", per_frame(parent)),
        Column::i64("world_index", per_frame(demo.world as i64)),
        Column::f32("timestamp", 1, None, (0..frames).map(|f| f as f32 / fps as f32).collect()),
        Column::i64("frame_index", (0..frames as i64).collect()),
        Column::i64("episode_index", per_frame(episode as i64)),
        Column::i64("index", (first_index as i64..(first_index + frames) as i64).collect()),
        Column::i64("task_index", per_frame(task as i64)),
    ]
}

/// The colour and depth images `camera` sees at every frame of `demo`.
fn image_columns(device: &Device, worlds: &Worlds, camera: &Camera, demo: &Demonstration) -> Result<[Column; 2]> {
    let frames = demo.trajectory.len();
    let moved: Vec<Option<(usize, Pose)>> =
        (0..frames).map(|f| demo.carried.as_ref().map(|c| (c.obstacle, c.poses[f]))).collect();
    let Images { width, height, rgb, depth } =
        device.render(worlds, camera, &vec![demo.world; frames], &demo.trajectory.positions, &moved)?;
    let pixels = (width * height) as usize;
    let millimetres: Vec<u16> = depth.iter().map(|&m| (m * 1000.0).round().clamp(0.0, 65535.0) as u16).collect();
    let encode = |color: png::ColorType, bit_depth: png::BitDepth, data: &[u8]| -> Result<Vec<u8>> {
        let mut out = vec![];
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(color);
        encoder.set_depth(bit_depth);
        let mut writer = encoder.write_header().map_err(|e| Error::Write(e.to_string()))?;
        writer.write_image_data(data).map_err(|e| Error::Write(e.to_string()))?;
        writer.finish().map_err(|e| Error::Write(e.to_string()))?;
        Ok(out)
    };
    let colour: Vec<Vec<u8>> = rgb
        .par_chunks(pixels * 3)
        .map(|frame| encode(png::ColorType::Rgb, png::BitDepth::Eight, frame))
        .collect::<Result<_>>()?;
    let depth_png: Vec<Vec<u8>> = millimetres
        .par_chunks(pixels)
        .map(|frame| {
            let big_endian: Vec<u8> = frame.iter().flat_map(|v| v.to_be_bytes()).collect();
            encode(png::ColorType::Grayscale, png::BitDepth::Sixteen, &big_endian)
        })
        .collect::<Result<_>>()?;
    let name = format!("observation.images.{}", camera.name);
    let image = |name: String, png, depth_map, stats| Column {
        name,
        width: width as usize,
        names: None,
        data: Data::Image { png, height, depth_map, stats: Box::new(stats) },
    };
    let rgb_stats = image_stats(frames, 3, 255.0, |c| rgb.iter().skip(c).step_by(3).map(|&v| v as usize));
    let depth_stats = image_stats(frames, 1, 1.0, |_| millimetres.iter().map(|&v| v as usize));
    Ok([image(name.clone(), colour, false, rgb_stats), image(format!("{name}_depth"), depth_png, true, depth_stats)])
}

/// Per-channel statistics of image pixels (`values(channel)`, integers), divided by `scale`, over
/// `frames` frames: exact quantiles (linear between order statistics) from a histogram.
fn image_stats<I: Iterator<Item = usize>>(
    frames: usize,
    channels: usize,
    scale: f64,
    values: impl Fn(usize) -> I,
) -> Stats {
    let mut s = Stats { count: frames, image: true, ..Default::default() };
    for c in 0..channels {
        let mut histogram = vec![0u64; 1 << 16];
        for v in values(c) {
            histogram[v] += 1;
        }
        let total: u64 = histogram.iter().sum();
        let (mut sum, mut squares) = (0.0, 0.0);
        for (v, &k) in histogram.iter().enumerate() {
            sum += k as f64 * v as f64;
            squares += k as f64 * (v as f64).powi(2);
        }
        let mean = sum / total as f64;
        let nth = |rank: u64| {
            let mut seen = 0;
            histogram
                .iter()
                .position(|&k| {
                    seen += k;
                    seen > rank
                })
                .unwrap_or(0) as f64
        };
        s.min.push(nth(0) / scale);
        s.max.push(nth(total - 1) / scale);
        s.mean.push(mean / scale);
        s.std.push((squares / total as f64 - mean * mean).max(0.0).sqrt() / scale);
        for (q, (level, _)) in s.quantiles.iter_mut().zip(QUANTILES) {
            let pos = level * (total - 1) as f64;
            let (lo, hi) = (nth(pos.floor() as u64), nth(pos.ceil() as u64));
            q.push((lo + (pos - pos.floor()) * (hi - lo)) / scale);
        }
    }
    s
}

/// The goal, then every obstacle of `world`, `carried` (an obstacle's index and pose) where it
/// is now.
fn environment_state(world: &World, goal: Pose, carried: Option<(usize, Pose)>, max_obstacles: usize) -> Vec<f32> {
    let canonical = |q: glam::Quat| if q.w < 0.0 { -q } else { q };
    let (p, q) = (goal.position, canonical(goal.rotation));
    let mut env = vec![p.x, p.y, p.z, q.x, q.y, q.z, q.w];
    for (k, o) in world.obstacles.iter().enumerate() {
        let o = match carried {
            Some((c, pose)) if c == k => o.placed(pose),
            _ => o.clone(),
        };
        let round = |radius: f32, half: f32| glam::Vec3::new(radius, radius, half);
        let (kind, center, size, rot) = match o {
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
    task: String,
    stats: Vec<Stats>,
}

/// Appends `rows`, episodes `first..`, to the episode-metadata files, each row recording the file
/// it lands in.
fn write_episodes(files: &mut Files, first: usize, rows: &[EpisodeRow], features: &[String]) -> Result<()> {
    let size = episodes_batch(first, rows, features, (0, 0))?.get_array_memory_size();
    let at = files.place(size)?;
    files.write(&episodes_batch(first, rows, features, at)?)
}

/// `meta/episodes` rows: boundaries into the data files plus flattened per-episode stats of each
/// of `features` (in column order), all in the episode-metadata file `at` (chunk, file).
fn episodes_batch(
    first: usize,
    episodes: &[EpisodeRow],
    features: &[String],
    at: (usize, usize),
) -> Result<RecordBatch> {
    let int_column = |f: &dyn Fn(usize, &EpisodeRow) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from(episodes.iter().enumerate().map(|(i, e)| f(i, e) as i64).collect::<Vec<_>>()))
    };
    let mut tasks = ListBuilder::new(StringBuilder::new());
    for e in episodes {
        tasks.values().append_value(&e.task);
        tasks.append(true);
    }
    let mut columns: Vec<(String, ArrayRef)> = vec![
        ("episode_index".into(), int_column(&|i, _| first + i)),
        ("tasks".into(), Arc::new(tasks.finish())),
        ("length".into(), int_column(&|_, e| e.length)),
        ("data/chunk_index".into(), int_column(&|_, e| e.chunk)),
        ("data/file_index".into(), int_column(&|_, e| e.file)),
        ("dataset_from_index".into(), int_column(&|_, e| e.from)),
        ("dataset_to_index".into(), int_column(&|_, e| e.from + e.length)),
        ("meta/episodes/chunk_index".into(), int_column(&|_, _| at.0)),
        ("meta/episodes/file_index".into(), int_column(&|_, _| at.1)),
    ];
    for (f, feature) in features.iter().enumerate() {
        let names: Vec<&str> = episodes[0].stats[f].vectors().iter().map(|(n, _)| *n).collect();
        for (s, stat) in names.iter().enumerate() {
            let array: ArrayRef = if episodes[0].stats[f].image {
                // `[channels][1][1]`, as LeRobot writes image statistics.
                let mut b = ListBuilder::new(ListBuilder::new(ListBuilder::new(Float64Builder::new())));
                for e in episodes {
                    for &v in e.stats[f].vectors()[s].1 {
                        b.values().values().values().append_value(v);
                        b.values().values().append(true);
                        b.values().append(true);
                    }
                    b.append(true);
                }
                Arc::new(b.finish())
            } else {
                let mut b = ListBuilder::new(Float64Builder::new());
                for e in episodes {
                    b.values().append_slice(e.stats[f].vectors()[s].1);
                    b.append(true);
                }
                Arc::new(b.finish())
            };
            columns.push((format!("stats/{feature}/{stat}"), array));
        }
        let mut count = ListBuilder::new(Int64Builder::new());
        for e in episodes {
            count.values().append_value(e.stats[f].count as i64);
            count.append(true);
        }
        columns.push((format!("stats/{feature}/count"), Arc::new(count.finish())));
    }
    let fields: Vec<Field> = columns.iter().map(|(n, a)| Field::new(n, a.data_type().clone(), true)).collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, a)| a).collect())?)
}

/// `meta/tasks.parquet` as pandas writes it: a `task_index` column indexed by the task string.
fn tasks_batch(tasks: &[&str]) -> Result<RecordBatch> {
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
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((0..tasks.len() as i64).collect::<Vec<_>>())),
        Arc::new(StringArray::from(tasks.to_vec())),
    ];
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

/// Parquet files under `dir` as `chunk-XXX/file-YYY.parquet`, one row group per write, starting a
/// new file once the current one would pass `limit` bytes and a new chunk every `CHUNKS_SIZE`
/// files.
struct Files {
    dir: PathBuf,
    limit: usize,
    chunk: usize,
    file: usize,
    bytes: usize,
    writer: Option<ArrowWriter<File>>,
    /// Bytes the next write takes, set by `place`.
    pending: usize,
}

impl Files {
    fn new(dir: PathBuf, limit: usize) -> Self {
        Self { dir, limit, chunk: 0, file: 0, bytes: 0, writer: None, pending: 0 }
    }

    /// The (chunk, file) the next write of `size` bytes lands in, starting a new file if needed.
    fn place(&mut self, size: usize) -> Result<(usize, usize)> {
        if self.writer.is_some() && self.bytes + size > self.limit {
            self.finish()?;
            self.file += 1;
            if self.file == CHUNKS_SIZE {
                (self.chunk, self.file) = (self.chunk + 1, 0);
            }
        }
        self.pending = size;
        Ok((self.chunk, self.file))
    }

    /// Writes `batch` where `place` put it.
    fn write(&mut self, batch: &RecordBatch) -> Result<()> {
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
        self.bytes += self.pending;
        Ok(())
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
