//! wgpu backend: one WGSL source runs on Vulkan (AMD, NVIDIA, Intel), Metal and DX12.

use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use bytemuck::{Pod, Zeroable};
use glam::Mat3;
use wgpu::util::DeviceExt;

use crate::device::{Backend, CollisionWeights, Evaluation, Worlds};
use crate::ik::IkOptions;
use crate::robot::{JointKind, MAX_DOF, MAX_JOINTS, MAX_LINKS, MAX_SPHERES, Robot};
use crate::sdf::SdfGrid;
use crate::trajopt::PlanOptions;
use crate::types::{JointPaths, Pose};
use crate::world::{Obstacle, World};

const WORKGROUP: u32 = 64;
/// Storage buffers in bind group 0: bindings 1..=9 are read-only, 10..=12 read-write (see kernels.wgsl).
const READ_ONLY_STORAGE: u32 = 9;
const READ_WRITE_STORAGE: u32 = 3;
/// Upper bounds on work per queue submission, to stay clear of driver watchdogs.
const EVAL_CHUNK: usize = 1 << 18;
const IK_ITERS_PER_SUBMIT: u32 = 16;
const TRAJ_ITERS_PER_SUBMIT: u32 = 25;

/// A `vec4<f32>` on the shader side.
type Vec4 = [f32; 4];

/// Declares a struct shared with the kernels once: the `#[repr(C)]` Rust struct and its WGSL
/// declaration come from the same field list, so host and shader layouts cannot drift apart.
/// Field types must also be WGSL type names (`u32`, `i32`, `f32` or `Vec4`).
macro_rules! shader_struct {
    ($(#[$doc:meta])* $rust:ident => $wgsl:ident { $($field:ident: $ty:ident),* $(,)? }) => {
        $(#[$doc])*
        #[repr(C)]
        #[derive(Clone, Copy, Default, Pod, Zeroable)]
        struct $rust {
            $($field: $ty),*
        }

        impl $rust {
            const WGSL: &'static str =
                concat!("struct ", stringify!($wgsl), " {\n", $("    ", stringify!($field), ": ", stringify!($ty), ",\n",)* "}\n");
        }
    };
}

shader_struct! {
    /// Per-call constants (`P` in kernels.wgsl).
    GpuParams => Params {
        n_dof: u32, n_links: u32, n_spheres: u32, n_pairs: u32,
        ee_link: u32, n_items: u32, points: u32, iterations: u32,
        w_world: f32, w_self: f32, margin: f32, self_margin: f32,
        w_acc: f32, w_vel: f32, beta1: f32, beta2: f32,
        damping: f32, rot_weight: f32, max_step: f32, collision_step: f32,
        samples: u32, pad0: u32, pad1: u32, pad2: u32,
    }
}

shader_struct! {
    /// One kinematic link: joint origin rotation columns and translation, joint axis, and
    /// `kind` 0 fixed / 1 revolute / 2 prismatic. A moving joint's value is
    /// `axis.w * q[dof] + trans.w` (multiplier and offset, for mimic joints), and `joint` numbers
    /// it among the moving joints. Bit `i` of `chain` marks link `i` as this link or an ancestor
    /// with a moving joint.
    GpuLink => Link {
        c0: Vec4, c1: Vec4, c2: Vec4, trans: Vec4, axis: Vec4,
        parent: i32, kind: u32, dof: u32, chain: u32, joint: u32, pad0: u32, pad1: u32, pad2: u32,
    }
}

shader_struct! {
    /// Collision sphere: center xyz and radius in `c`.
    GpuSphere => Sphere { c: Vec4, link: u32, self_buf: f32, pad0: u32, pad1: u32 }
}

/// Obstacle kinds as stored in `GpuObstacle::center.w`.
const CUBOID: u32 = 0;
const SPHERE: u32 = 1;
const CYLINDER: u32 = 2;
const CAPSULE: u32 = 3;
const SDF: u32 = 4;

shader_struct! {
    /// `center.w` is the kind (`CUBOID`, `SPHERE`, ...). `half` holds the half extents of a cuboid,
    /// the radius (x) and half height or half length (y) of the round kinds, or a distance grid's
    /// origin (xyz) and voxel size (w). r0..r2 are the world-from-local rotation columns. A grid
    /// has `nx * ny * nz` points, two per `grid_data` word from word `grid`.
    GpuObstacle => Obstacle {
        center: Vec4, half: Vec4, r0: Vec4, r1: Vec4, r2: Vec4,
        nx: u32, ny: u32, nz: u32, grid: u32,
    }
}

shader_struct! {
    /// Adam schedule for one trajopt iteration (`IT` in kernels.wgsl).
    GpuIter => Iter { lr: f32, bc1: f32, bc2: f32, eps: f32 }
}

fn v4(v: glam::Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

pub(crate) struct GpuBackend {
    robot: Robot,
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    evaluate: wgpu::ComputePipeline,
    ik: wgpu::ComputePipeline,
    traj_samples: wgpu::ComputePipeline,
    traj_grad: wgpu::ComputePipeline,
    traj_update: wgpu::ComputePipeline,
    links: wgpu::Buffer,
    spheres: wgpu::Buffer,
    pairs: wgpu::Buffer,
    limits: wgpu::Buffer,
    uniform_align: u64,
}

/// Worlds in device memory: every world's obstacles, each world's `[first, count]` range of them,
/// and the distance grids they use, each stored once.
struct GpuWorlds {
    obstacles: wgpu::Buffer,
    ranges: wgpu::Buffer,
    grids: wgpu::Buffer,
}

/// Per-call buffers bound alongside the robot buffers.
struct CallBuffers<'a> {
    params: &'a wgpu::Buffer,
    worlds: &'a GpuWorlds,
    item_world: &'a wgpu::Buffer,
    targets: &'a wgpu::Buffer,
    q: &'a wgpu::Buffer,
    aux: &'a wgpu::Buffer,
    out: &'a wgpu::Buffer,
}

impl GpuBackend {
    /// Uses the first adapter whose name contains `name` (any adapter if `None`), preferring
    /// discrete, then integrated GPUs. OpenGL adapters are skipped.
    pub(crate) fn new(robot: &Robot, name: Option<&str>) -> Result<Self> {
        if robot.dof() > MAX_DOF || robot.spheres.len() > MAX_SPHERES || robot.links.len() > MAX_LINKS {
            bail!("robot exceeds kernel limits ({MAX_DOF} joints, {MAX_SPHERES} spheres, {MAX_LINKS} links)");
        }
        let wanted = name.map(str::to_lowercase);
        let accept = |info: &wgpu::AdapterInfo| {
            info.backend != wgpu::Backend::Gl && wanted.as_ref().is_none_or(|w| info.name.to_lowercase().contains(w))
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let rank = |t: wgpu::DeviceType| match t {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Cpu => 3,
            wgpu::DeviceType::Other => 4,
        };
        let mut adapters: Vec<_> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .filter(|a| accept(&a.get_info()))
            .collect();
        adapters.sort_by_key(|a| rank(a.get_info().device_type));
        let mut rejected = vec![];
        let adapter = adapters
            .into_iter()
            .find(|a| {
                let unmet = unmet_limits(&a.limits());
                if !unmet.is_empty() {
                    let info = a.get_info();
                    rejected.push(format!("{} ({:?}) lacks {}", info.name, info.backend, unmet.join(", ")));
                }
                unmet.is_empty()
            })
            .ok_or_else(|| match (&rejected[..], name) {
                ([], Some(name)) => anyhow!("no Vulkan, Metal or DX12 adapter named like '{name}'"),
                ([], None) => anyhow!("no Vulkan, Metal or DX12 adapter found"),
                _ => anyhow!("no GPU adapter can run the kernels:\n  {}", rejected.join("\n  ")),
            })?;
        let info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("batchplan"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("requesting GPU device")?;

        // Constants and shared structs are generated from their Rust definitions.
        let source = [
            "alias Vec4 = vec4<f32>;\n",
            &format!("const MAX_DOF: u32 = {MAX_DOF}u;\nconst MAX_LINKS: u32 = {MAX_LINKS}u;\n"),
            &format!("const MAX_JOINTS: u32 = {MAX_JOINTS}u;\n"),
            &format!("const MAX_SPHERES: u32 = {MAX_SPHERES}u;\nconst JAC_LEN: u32 = {}u;\n", 6 * MAX_DOF),
            &format!("const CUBOID: u32 = {CUBOID}u;\nconst SPHERE: u32 = {SPHERE}u;\n"),
            &format!(
                "const CYLINDER: u32 = {CYLINDER}u;\nconst CAPSULE: u32 = {CAPSULE}u;\nconst SDF: u32 = {SDF}u;\n"
            ),
            GpuParams::WGSL,
            GpuLink::WGSL,
            GpuSphere::WGSL,
            GpuObstacle::WGSL,
            GpuIter::WGSL,
            include_str!("kernels.wgsl"),
        ]
        .concat();
        // Turn shader and pipeline validation failures into errors instead of panics.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batchplan kernels"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let buffer_entry = |binding: u32, ty: wgpu::BufferBindingType, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: None },
            count: None,
        };
        let mut entries = vec![buffer_entry(0, wgpu::BufferBindingType::Uniform, false)];
        let (ro, rw) = (1..=READ_ONLY_STORAGE, READ_ONLY_STORAGE + 1..=READ_ONLY_STORAGE + READ_WRITE_STORAGE);
        entries.extend(ro.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: true }, false)));
        entries.extend(rw.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: false }, false)));
        let layout0 = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("main"), entries: &entries });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iteration"),
            entries: &[buffer_entry(0, wgpu::BufferBindingType::Uniform, true)],
        });
        let pl_main = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout0)],
            immediate_size: 0,
        });
        let pl_iter = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout0), Some(&layout1)],
            immediate_size: 0,
        });
        let pipeline = |layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let evaluate = pipeline(&pl_main, "evaluate_main");
        let ik = pipeline(&pl_main, "ik_main");
        let traj_samples = pipeline(&pl_iter, "traj_samples");
        let traj_grad = pipeline(&pl_iter, "traj_grad");
        let traj_update = pipeline(&pl_iter, "traj_update");
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("{} ({:?}) cannot build the kernels: {e}", info.name, info.backend);
        }

        let mut moving = 0;
        let gpu_links: Vec<GpuLink> = robot
            .links
            .iter()
            .map(|l| {
                let joint = moving;
                moving += u32::from(l.joint.actuation().is_some());
                let (kind, dof, axis, multiplier, offset) = match l.joint {
                    JointKind::Fixed => (0, 0, glam::Vec3::ZERO, 0.0, 0.0),
                    JointKind::Revolute { dof, axis, multiplier, offset } => (1, dof as u32, axis, multiplier, offset),
                    JointKind::Prismatic { dof, axis, multiplier, offset } => (2, dof as u32, axis, multiplier, offset),
                };
                GpuLink {
                    c0: v4(l.origin.rot.x_axis, 0.0),
                    c1: v4(l.origin.rot.y_axis, 0.0),
                    c2: v4(l.origin.rot.z_axis, 0.0),
                    trans: v4(l.origin.trans, offset),
                    axis: v4(axis, multiplier),
                    parent: l.parent.map_or(-1, |p| p as i32),
                    kind,
                    dof,
                    chain: l.chain,
                    joint,
                    ..Default::default()
                }
            })
            .collect();
        let gpu_spheres: Vec<GpuSphere> = robot
            .spheres
            .iter()
            .map(|s| GpuSphere {
                c: v4(s.center, s.radius),
                link: s.link as u32,
                self_buf: s.self_buffer,
                ..Default::default()
            })
            .collect();
        let gpu_limits: Vec<[f32; 2]> = (0..robot.dof()).map(|j| [robot.lower[j], robot.upper[j]]).collect();

        Ok(Self {
            robot: robot.clone(),
            info,
            links: storage(&device, "links", &gpu_links),
            spheres: storage(&device, "spheres", &gpu_spheres),
            pairs: storage(&device, "pairs", &robot.self_pairs),
            limits: storage(&device, "limits", &gpu_limits),
            uniform_align: u64::from(adapter_limits.min_uniform_buffer_offset_alignment).max(16),
            device,
            queue,
            layout0,
            layout1,
            evaluate,
            ik,
            traj_samples,
            traj_grad,
            traj_update,
        })
    }

    fn params(&self, n_items: usize, w: &CollisionWeights) -> GpuParams {
        GpuParams {
            n_dof: self.robot.dof() as u32,
            n_links: self.robot.links.len() as u32,
            n_spheres: self.robot.spheres.len() as u32,
            n_pairs: self.robot.self_pairs.len() as u32,
            ee_link: self.robot.ee_link as u32,
            n_items: n_items as u32,
            w_world: w.world,
            w_self: w.self_collision,
            margin: w.margin,
            self_margin: w.self_margin,
            ..Default::default()
        }
    }

    fn upload_worlds(&self, worlds: &[World]) -> Result<GpuWorlds> {
        let mut obstacles = vec![];
        let mut ranges: Vec<[u32; 2]> = vec![];
        let mut grid_words: Vec<u32> = vec![];
        let mut grid_offsets: HashMap<*const SdfGrid, u32> = HashMap::new();
        for s in worlds {
            ranges.push([obstacles.len() as u32, s.obstacles.len() as u32]);
            for o in &s.obstacles {
                let rotated = |kind: u32, center: glam::Vec3, half: [f32; 4], rotation: glam::Quat| {
                    let r = Mat3::from_quat(rotation);
                    GpuObstacle {
                        center: v4(center, kind as f32),
                        half,
                        r0: v4(r.x_axis, 0.0),
                        r1: v4(r.y_axis, 0.0),
                        r2: v4(r.z_axis, 0.0),
                        ..Default::default()
                    }
                };
                obstacles.push(match *o {
                    Obstacle::Cuboid { center, half_extents, rotation } => {
                        rotated(CUBOID, center, v4(half_extents, 0.0), rotation)
                    }
                    Obstacle::Sphere { center, radius } => {
                        rotated(SPHERE, center, [radius, 0.0, 0.0, 0.0], glam::Quat::IDENTITY)
                    }
                    Obstacle::Cylinder { center, rotation, radius, half_height } => {
                        rotated(CYLINDER, center, [radius, half_height, 0.0, 0.0], rotation)
                    }
                    Obstacle::Capsule { center, rotation, radius, half_length } => {
                        rotated(CAPSULE, center, [radius, half_length, 0.0, 0.0], rotation)
                    }
                    Obstacle::Sdf { ref grid, center, rotation } => {
                        let offset =
                            *grid_offsets.entry(Arc::as_ptr(grid)).or_insert_with(|| {
                                let offset = grid_words.len() as u32;
                                grid_words.extend(grid.values.chunks(2).map(|pair| {
                                    u32::from(pair[0]) | u32::from(pair.get(1).copied().unwrap_or(0)) << 16
                                }));
                                offset
                            });
                        let [nx, ny, nz] = grid.dims;
                        GpuObstacle {
                            nx,
                            ny,
                            nz,
                            grid: offset,
                            ..rotated(SDF, center, v4(grid.origin, grid.voxel), rotation)
                        }
                    }
                });
            }
        }
        let limits = self.device.limits();
        let bytes = (grid_words.len() * 4) as u64;
        ensure!(
            bytes <= limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
            "the distance grids take {} MiB, more than {} can bind",
            bytes >> 20,
            self.info.name
        );
        Ok(GpuWorlds {
            obstacles: storage(&self.device, "obstacles", &obstacles),
            ranges: storage(&self.device, "world ranges", &ranges),
            grids: storage(&self.device, "distance grids", &grid_words),
        })
    }

    fn uniform(&self, params: &GpuParams) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn bind_main(&self, b: &CallBuffers) -> wgpu::BindGroup {
        let buffers = [
            b.params,
            &self.links,
            &self.spheres,
            &self.pairs,
            &self.limits,
            &b.worlds.obstacles,
            &b.worlds.ranges,
            b.item_world,
            b.targets,
            &b.worlds.grids,
            b.q,
            b.aux,
            b.out,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry { binding: i as u32, resource: buf.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout0,
            entries: &entries,
        })
    }

    fn submit_pass(&self, f: impl FnOnce(&mut wgpu::ComputePass)) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            f(&mut pass);
        }
        self.queue.submit([enc.finish()]);
    }

    fn read(&self, buf: &wgpu::Buffer, floats: usize) -> Result<Vec<f32>> {
        let size = (floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()?).to_vec();
        staging.unmap();
        Ok(out)
    }

    fn evaluate_chunk(
        &self,
        worlds: &GpuWorlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Vec<f32>> {
        let items = item_world.len();
        let stride = 3 + self.robot.dof();
        let params = self.uniform(&self.params(items, w));
        let world_buf = storage(&self.device, "item world", item_world);
        let no_targets = storage::<[f32; 4]>(&self.device, "unused targets", &[]);
        let no_aux = storage::<f32>(&self.device, "unused aux", &[]);
        let q_buf = storage(&self.device, "q", q);
        let out = storage_zeroed(&self.device, "out", items * stride);
        let bg = self.bind_main(&CallBuffers {
            params: &params,
            worlds,
            item_world: &world_buf,
            targets: &no_targets,
            q: &q_buf,
            aux: &no_aux,
            out: &out,
        });
        self.submit_pass(|pass| {
            pass.set_pipeline(&self.evaluate);
            pass.set_bind_group(0, &bg, &[]);
            dispatch(pass, items);
        });
        self.read(&out, items * stride)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String {
        format!("gpu ({}, {:?}, {})", self.info.name, self.info.backend, self.info.driver_info)
    }

    fn robot(&self) -> &Robot {
        &self.robot
    }

    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(self.upload_worlds(worlds)?))
    }

    fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation> {
        let worlds: &GpuWorlds = worlds.prepared();
        let n = self.robot.dof();
        let stride = 3 + n;
        let mut out = Evaluation::default();
        for (chunk_world, chunk_q) in item_world.chunks(EVAL_CHUNK).zip(q.chunks(EVAL_CHUNK * n)) {
            let raw = self.evaluate_chunk(worlds, chunk_world, chunk_q, w)?;
            for row in raw.chunks(stride) {
                out.world_clearance.push(row[0]);
                out.self_clearance.push(row[1]);
                out.cost.push(row[2]);
                out.grad.extend_from_slice(&row[3..]);
            }
        }
        Ok(out)
    }

    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        let worlds: &GpuWorlds = worlds.prepared();
        let items = item_world.len();
        if items == 0 {
            return Ok(vec![]);
        }
        let mut params = self.params(items, &o.collision);
        params.damping = o.damping;
        params.rot_weight = o.rot_weight;
        params.max_step = o.max_step;
        params.collision_step = o.collision_step;
        let params_buf = self.uniform(&params);
        let world_buf = storage(&self.device, "item world", item_world);
        let target_data: Vec<[f32; 4]> = targets
            .iter()
            .flat_map(|t| {
                let r = Mat3::from_quat(t.rotation);
                [v4(t.position, 0.0), v4(r.x_axis, 0.0), v4(r.y_axis, 0.0), v4(r.z_axis, 0.0)]
            })
            .collect();
        let target_buf = storage(&self.device, "targets", &target_data);
        let dummy = storage::<f32>(&self.device, "unused", &[]);
        let q_buf = storage(&self.device, "q", q);
        let out = storage_zeroed(&self.device, "ik errors", items * 2);
        let bg = self.bind_main(&CallBuffers {
            params: &params_buf,
            worlds,
            item_world: &world_buf,
            targets: &target_buf,
            q: &q_buf,
            aux: &dummy,
            out: &out,
        });
        let mut done = 0;
        while done < o.iterations {
            params.iterations = IK_ITERS_PER_SUBMIT.min(o.iterations - done);
            self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            self.submit_pass(|pass| {
                pass.set_pipeline(&self.ik);
                pass.set_bind_group(0, &bg, &[]);
                dispatch(pass, items);
            });
            done += params.iterations;
        }
        if o.iterations == 0 {
            params.iterations = 0;
            self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            self.submit_pass(|pass| {
                pass.set_pipeline(&self.ik);
                pass.set_bind_group(0, &bg, &[]);
                dispatch(pass, items);
            });
        }
        q.copy_from_slice(&self.read(&q_buf, q.len())?);
        Ok(self.read(&out, items * 2)?.chunks(2).map(|e| [e[0], e[1]]).collect())
    }

    fn trajopt(&self, worlds: &Worlds, item_world: &[u32], paths: &mut JointPaths, o: &PlanOptions) -> Result<()> {
        let worlds: &GpuWorlds = worlds.prepared();
        let traj = &mut paths.positions[..];
        let items = item_world.len();
        if items == 0 || o.iterations == 0 {
            return Ok(());
        }
        let mut params = self.params(items, &o.collision);
        params.points = paths.points as u32;
        params.samples = o.samples_per_span as u32;
        params.w_acc = o.w_acc;
        params.w_vel = o.w_vel;
        params.beta1 = o.beta1;
        params.beta2 = o.beta2;
        let params_buf = self.uniform(&params);
        let world_buf = storage(&self.device, "item world", item_world);
        let dummy = storage::<f32>(&self.device, "unused", &[]);
        let q_buf = storage(&self.device, "trajectories", traj);
        // [collision gradient per sample | gradient per control point | Adam m | Adam v]
        let sample_threads = items * (paths.points - 3) * o.samples_per_span;
        let aux = storage_zeroed(&self.device, "trajopt state", sample_threads * paths.dof + traj.len() * 3);
        let out = storage::<f32>(&self.device, "unused out", &[]);
        let bg = self.bind_main(&CallBuffers {
            params: &params_buf,
            worlds,
            item_world: &world_buf,
            targets: &dummy,
            q: &q_buf,
            aux: &aux,
            out: &out,
        });
        let align = self.uniform_align as usize;
        let mut schedule = vec![0u8; align * o.iterations as usize];
        for k in 0..o.iterations {
            let [lr, bc1, bc2] = o.schedule(k);
            let it = GpuIter { lr, bc1, bc2, eps: o.adam_epsilon };
            schedule[k as usize * align..k as usize * align + 16].copy_from_slice(bytemuck::bytes_of(&it));
        }
        let iter_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("schedule"),
            contents: &schedule,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg_iter = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &iter_buf,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
            }],
        });
        let free_threads = items * (paths.points - 6);
        let mut k = 0;
        while k < o.iterations {
            let end = (k + TRAJ_ITERS_PER_SUBMIT).min(o.iterations);
            self.submit_pass(|pass| {
                pass.set_bind_group(0, &bg, &[]);
                for it in k..end {
                    let offset = it * self.uniform_align as u32;
                    pass.set_bind_group(1, &bg_iter, &[offset]);
                    pass.set_pipeline(&self.traj_samples);
                    dispatch(pass, sample_threads);
                    pass.set_pipeline(&self.traj_grad);
                    dispatch(pass, free_threads);
                    pass.set_pipeline(&self.traj_update);
                    dispatch(pass, free_threads);
                }
            });
            k = end;
        }
        traj.copy_from_slice(&self.read(&q_buf, traj.len())?);
        Ok(())
    }
}

/// Adapter limits the kernels need beyond what wgpu guarantees; empty when the adapter suffices.
fn unmet_limits(l: &wgpu::Limits) -> Vec<String> {
    let storage = READ_ONLY_STORAGE + READ_WRITE_STORAGE;
    let mut unmet = vec![];
    if l.max_storage_buffers_per_shader_stage < storage {
        unmet.push(format!(
            "{storage} storage buffers per shader stage (has {})",
            l.max_storage_buffers_per_shader_stage
        ));
    }
    if l.max_bind_groups < 2 {
        unmet.push(format!("2 bind groups (has {})", l.max_bind_groups));
    }
    if l.max_compute_workgroup_size_x < WORKGROUP || l.max_compute_invocations_per_workgroup < WORKGROUP {
        unmet.push(format!("{WORKGROUP}-wide compute workgroups"));
    }
    unmet
}

fn dispatch(pass: &mut wgpu::ComputePass, threads: usize) {
    let groups = (threads as u32).div_ceil(WORKGROUP);
    if groups == 0 {
        return;
    }
    let x = groups.min(65535);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

/// A storage buffer holding `data`, padded to at least one element: WGSL requires a binding to
/// hold one element of its array type even when the batch has none (e.g. a world with no obstacles).
fn storage<T: Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    let mut bytes = bytemuck::cast_slice(data).to_vec();
    bytes.resize(bytes.len().max(std::mem::size_of::<T>()).max(16), 0);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    })
}

fn storage_zeroed(device: &wgpu::Device, label: &str, floats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (floats.max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_webgpu_limits_are_rejected_with_the_reason() {
        let unmet = unmet_limits(&wgpu::Limits::defaults());
        assert_eq!(unmet, ["12 storage buffers per shader stage (has 8)"]);
    }
}
