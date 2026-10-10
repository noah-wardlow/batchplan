//! wgpu backend: one WGSL source runs on Vulkan (AMD, NVIDIA, Intel), Metal and DX12.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::error::{Error, Result};
use bytemuck::{Pod, Zeroable};
use glam::Mat3;
use wgpu::util::DeviceExt;

use crate::device::{Backend, CollisionWeights, Evaluation, Worlds};
use crate::ik::IkOptions;
use crate::robot::{JointKind, Robot};
use crate::sdf::SdfGrid;
use crate::trajopt::{LINE_SEARCH, MAX_HISTORY, PlanOptions};
use crate::types::{JointPaths, Pose};
use crate::world::{Obstacle, World};

const WORKGROUP: u32 = 64;
/// Storage buffers in bind group 0: bindings 1..=9 are read-only, 10..=12 read-write (see kernels.wgsl).
const READ_ONLY_STORAGE: u32 = 9;
const READ_WRITE_STORAGE: u32 = 3;
/// Upper bounds on work per queue submission, to stay clear of driver watchdogs.
const EVAL_CHUNK: usize = 1 << 18;
const IK_ITERS_PER_SUBMIT: u32 = 16;
const IK_CHUNK: usize = 1 << 16;
const TRAJ_ITERS_PER_SUBMIT: u32 = 8;
const TRAJ_CHUNK: usize = 1 << 13;

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
        n_dof: u32, n_items: u32, points: u32, iterations: u32,
        w_world: f32, w_self: f32, margin: f32, self_margin: f32,
        w_acc: f32, w_vel: f32, initial_step: f32, history: u32,
        damping: f32, rot_weight: f32, max_step: f32, collision_step: f32,
        samples: u32, pad0: u32, pad1: u32, pad2: u32,
    }
}

shader_struct! {
    /// One kinematic link: joint origin rotation columns and translation, and joint axis. A moving
    /// joint's value is `axis.w * q[dof] + trans.w` (multiplier and offset, for mimic joints).
    /// `bound` is the sphere (center in the link frame, radius) around the link's collision spheres
    /// and their self-collision buffers, which are spheres `first_sphere..first_sphere + n_spheres`.
    /// The tree itself is written into the generated kernels (`robot_wgsl`).
    GpuLink => Link {
        c0: Vec4, c1: Vec4, c2: Vec4, trans: Vec4, axis: Vec4, bound: Vec4,
        first_sphere: u32, n_spheres: u32, pad0: u32, pad1: u32,
    }
}

shader_struct! {
    /// Collision sphere: center xyz and radius in `c`.
    GpuSphere => Sphere { c: Vec4, self_buf: f32, pad0: u32, pad1: u32, pad2: u32 }
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

/// The robot-specific part of the kernels: forward kinematics, the end effector's Jacobian and
/// the collision cost, written out link by link. Each link's frame (`rot_i`, `pos_i`) and collision
/// wrench (`force_i`, `moment_i`, about the world origin) is its own private variable indexed only
/// by constants, so it stays in registers; private arrays indexed at run time end up in scratch
/// memory on some drivers. Sphere and sphere-pair ranges are read from the buffers, so robots that
/// differ only in their spheres share kernels.
fn robot_wgsl(robot: &Robot) -> String {
    use std::fmt::Write;
    let links = &robot.links;
    let (dof, ee) = (robot.dof(), robot.ee_link);
    let mut w = format!("const MAX_DOF: u32 = {dof}u;\nconst JAC_LEN: u32 = {}u;\n", 6 * dof);
    for i in 0..links.len() {
        let _ = writeln!(w, "var<private> rot_{i}: mat3x3<f32>;\nvar<private> pos_{i}: vec3<f32>;");
        let _ = writeln!(w, "var<private> force_{i}: vec3<f32>;\nvar<private> moment_{i}: vec3<f32>;");
    }

    w += "\n// Every link's frame, root first.\nfn fk() {\n";
    for (i, l) in links.iter().enumerate() {
        let (prot, ppos) =
            l.parent.map_or((String::new(), String::new()), |p| (format!("rot_{p} * "), format!(" + pos_{p}")));
        let _ = writeln!(w, "    {{\n        let l = links[{i}u];");
        let _ = writeln!(w, "        let jrot = {prot}mat3x3<f32>(l.c0.xyz, l.c1.xyz, l.c2.xyz);");
        let _ = writeln!(w, "        let jpos = {prot}l.trans.xyz{ppos};");
        let _ = match l.joint {
            JointKind::Revolute { dof, .. } => writeln!(
                w,
                "        rot_{i} = jrot * rodrigues(l.axis.xyz, l.axis.w * q[{dof}u] + l.trans.w);\n        pos_{i} = jpos;"
            ),
            JointKind::Prismatic { dof, .. } => writeln!(
                w,
                "        rot_{i} = jrot;\n        pos_{i} = jpos + jrot * l.axis.xyz * (l.axis.w * q[{dof}u] + l.trans.w);"
            ),
            JointKind::Fixed => writeln!(w, "        rot_{i} = jrot;\n        pos_{i} = jpos;"),
        };
        w += "    }\n";
    }
    w += "}\n";

    let _ = writeln!(w, "\nfn ee_pos() -> vec3<f32> {{\n    return pos_{ee};\n}}");
    let _ = writeln!(w, "\nfn ee_rot() -> mat3x3<f32> {{\n    return rot_{ee};\n}}");
    w += "\n// The end effector's Jacobian into `jac` (6 x MAX_DOF, row-major; rotation rows scaled by\n";
    w += "// P.rot_weight), its moving joints root first, mimic joints adding into their leader.\n";
    w += "fn ee_jacobian() {\n    for (var k = 0u; k < JAC_LEN; k++) {\n        jac[k] = 0.0;\n    }\n";
    for (i, d, _) in robot.chain(ee) {
        let _ =
            writeln!(w, "    {{\n        let m = links[{i}u].axis.w;\n        let a = rot_{i} * links[{i}u].axis.xyz;");
        if matches!(links[i].joint, JointKind::Prismatic { .. }) {
            w += "        let jp = m * a;\n        let jo = vec3<f32>(0.0);\n";
        } else {
            let _ = writeln!(
                w,
                "        let jp = m * cross(a, pos_{ee} - pos_{i});\n        let jo = m * a * P.rot_weight;"
            );
        }
        for (r, c) in ["jp.x", "jp.y", "jp.z", "jo.x", "jo.y", "jo.z"].iter().enumerate() {
            let _ = writeln!(w, "        jac[{r}u * MAX_DOF + {d}u] += {c};");
        }
        w += "    }\n";
    }
    w += "}\n";

    w += r"
// Collision cost of the configuration last passed to fk(); with `gradient`, adds d(cost)/dq into
// `grad`. Returns (cost, world clearance, self clearance). Callers pass `gradient` as a constant,
// so the compiler drops the gradient work from cost-only kernels.
fn collision(world: u32, w_world: f32, w_self: f32, margin: f32, self_margin: f32, gradient: bool) -> vec3<f32> {
    var cost = 0.0;
    var wmin = FAR;
    var smin = FAR;
    var touched = false;
    let range = world_ranges[world];
    if (gradient) {
";
    for i in 0..links.len() {
        let _ = writeln!(w, "        force_{i} = vec3<f32>(0.0);\n        moment_{i} = vec3<f32>(0.0);");
    }
    w += "    }\n";
    w += "    // An obstacle farther from a link's bounding sphere than the margin cannot add cost through\n";
    w += "    // the link's spheres; the gap bounds their clearance from below. Distance grids are\n";
    w += "    // interpolated, not exact, so their spheres are always checked.\n";
    w += "    let world_gate = max(margin, 0.0);\n";
    for i in (0..links.len()).filter(|&i| robot.sphere_ranges[i][1] > 0) {
        let _ = write!(
            w,
            r"    {{
        let bl = links[{i}u].bound;
        let bc = rot_{i} * bl.xyz + pos_{i};
        for (var k = 0u; k < range.y; k++) {{
            let o = obstacles[range.x + k];
            if (u32(o.center.w) != SDF) {{
                let gap = obstacle_distance(o, bc).w - bl.w;
                if (gap > world_gate) {{
                    wmin = min(wmin, gap);
                    continue;
                }}
            }}
            for (var s = links[{i}u].first_sphere; s < links[{i}u].first_sphere + links[{i}u].n_spheres; s++) {{
                let sp = spheres[s];
                let c = rot_{i} * sp.c.xyz + pos_{i};
                let dg = obstacle_distance(o, c);
                let d = dg.w - sp.c.w;
                wmin = min(wmin, d);
                let pen = margin - d;
                if (pen > 0.0) {{
                    cost += w_world * pen * pen;
                    if (gradient) {{
                        let f = -2.0 * w_world * pen * dg.xyz;
                        force_{i} += f;
                        moment_{i} += cross(c, f);
                        touched = true;
                    }}
                }}
            }}
        }}
    }}
"
        );
    }
    w += "    // Likewise for link pairs whose bounding spheres are farther apart than the self margin.\n";
    w += "    let gate = max(self_margin, 0.0);\n";
    for (l, lp) in robot.self_link_pairs.iter().enumerate() {
        let (a, b) = (lp.a, lp.b);
        let _ = write!(
            w,
            r"    {{
        let ba = links[{a}u].bound;
        let bb = links[{b}u].bound;
        let gap = length(rot_{a} * ba.xyz + pos_{a} - rot_{b} * bb.xyz - pos_{b}) - ba.w - bb.w;
        if (gap > gate) {{
            smin = min(smin, gap);
        }} else {{
            let span = pairs[{l}u];
            for (var k = span.x; k < span.x + span.y; k++) {{
                let pr = pairs[k];
                let sa = spheres[pr.x];
                let sb = spheres[pr.y];
                let ca = rot_{a} * sa.c.xyz + pos_{a};
                let cb = rot_{b} * sb.c.xyz + pos_{b};
                let diff = ca - cb;
                let dist = length(diff);
                let d = dist - (sa.c.w + sa.self_buf) - (sb.c.w + sb.self_buf);
                smin = min(smin, d);
                let pen = self_margin - d;
                if (pen > 0.0) {{
                    cost += w_self * pen * pen;
                    if (gradient) {{
                        var u = vec3<f32>(1.0, 0.0, 0.0);
                        if (dist > 1e-9) {{
                            u = diff / dist;
                        }}
                        let g = 2.0 * w_self * pen * u;
                        force_{a} -= g;
                        moment_{a} -= cross(ca, g);
                        force_{b} += g;
                        moment_{b} += cross(cb, g);
                        touched = true;
                    }}
                }}
            }}
        }}
    }}
"
        );
    }
    w += "    // Each link's wrench reaches the joints at and above it, leaves first.\n";
    w += "    if (gradient && touched) {\n";
    for (i, l) in links.iter().enumerate().rev() {
        let _ = match l.joint {
            JointKind::Revolute { dof, .. } => writeln!(
                w,
                "        grad[{dof}u] += links[{i}u].axis.w * dot(rot_{i} * links[{i}u].axis.xyz, moment_{i} - cross(pos_{i}, force_{i}));"
            ),
            JointKind::Prismatic { dof, .. } => writeln!(
                w,
                "        grad[{dof}u] += links[{i}u].axis.w * dot(rot_{i} * links[{i}u].axis.xyz, force_{i});"
            ),
            JointKind::Fixed => Ok(()),
        };
        if let Some(p) = l.parent {
            let _ = writeln!(w, "        force_{p} += force_{i};\n        moment_{p} += moment_{i};");
        }
    }
    w += "    }\n    return vec3<f32>(cost, wmin, smin);\n}\n";
    w
}

fn v4(v: glam::Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

#[derive(Clone)]
pub(crate) struct GpuBackend {
    robot: Robot,
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Arc<Kernels>,
    /// Kernels built on this device, by their robot-specific source. Every backend derived with
    /// `with_robot` shares it, so attaching and detaching objects compiles each robot shape once.
    built: Arc<Mutex<HashMap<String, Arc<Kernels>>>>,
    buffers: RobotBuffers,
}

/// The compiled kernels for one robot shape.
struct Kernels {
    layout0: wgpu::BindGroupLayout,
    evaluate: wgpu::ComputePipeline,
    ik: wgpu::ComputePipeline,
    traj_costs: wgpu::ComputePipeline,
    traj_search: wgpu::ComputePipeline,
    traj_samples: wgpu::ComputePipeline,
    traj_grad: wgpu::ComputePipeline,
    lbfgs_direction: wgpu::ComputePipeline,
}

/// The robot as the kernels read it.
#[derive(Clone)]
struct RobotBuffers {
    links: wgpu::Buffer,
    spheres: wgpu::Buffer,
    pairs: wgpu::Buffer,
    limits: wgpu::Buffer,
}

impl RobotBuffers {
    fn new(device: &wgpu::Device, robot: &Robot) -> Self {
        let gpu_links: Vec<GpuLink> = robot
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let (axis, multiplier, offset) = match l.joint {
                    JointKind::Fixed => (glam::Vec3::ZERO, 0.0, 0.0),
                    JointKind::Revolute { axis, multiplier, offset, .. }
                    | JointKind::Prismatic { axis, multiplier, offset, .. } => (axis, multiplier, offset),
                };
                GpuLink {
                    c0: v4(l.origin.rot.x_axis, 0.0),
                    c1: v4(l.origin.rot.y_axis, 0.0),
                    c2: v4(l.origin.rot.z_axis, 0.0),
                    trans: v4(l.origin.trans, offset),
                    axis: v4(axis, multiplier),
                    bound: robot.link_bounds[i],
                    first_sphere: robot.sphere_ranges[i][0],
                    n_spheres: robot.sphere_ranges[i][1],
                    ..Default::default()
                }
            })
            .collect();
        let gpu_spheres: Vec<GpuSphere> = robot
            .spheres
            .iter()
            .map(|s| GpuSphere { c: v4(s.center, s.radius), self_buf: s.self_buffer, ..Default::default() })
            .collect();
        // Each link pair's (first, count) of the sphere pairs that follow.
        let skip = robot.self_link_pairs.len() as u32;
        let gpu_pairs: Vec<[u32; 2]> = robot
            .self_link_pairs
            .iter()
            .map(|lp| [skip + lp.first, lp.count])
            .chain(robot.self_pairs.iter().copied())
            .collect();
        let gpu_limits: Vec<[f32; 2]> = (0..robot.dof()).map(|j| [robot.lower[j], robot.upper[j]]).collect();

        Self {
            links: storage(device, "links", &gpu_links),
            spheres: storage(device, "spheres", &gpu_spheres),
            pairs: storage(device, "pairs", &gpu_pairs),
            limits: storage(device, "limits", &gpu_limits),
        }
    }
}

/// The kernels for `robot`'s shape, built on `device` unless `built` already holds them.
fn kernels_for(
    device: &wgpu::Device,
    info: &wgpu::AdapterInfo,
    built: &Mutex<HashMap<String, Arc<Kernels>>>,
    robot: &Robot,
) -> Result<Arc<Kernels>> {
    let robot_source = robot_wgsl(robot);
    let mut built = built.lock().expect("no thread panics while holding the kernel cache");
    if let Some(kernels) = built.get(&robot_source) {
        return Ok(kernels.clone());
    }
    let kernels = Arc::new(build_kernels(device, info, &robot_source)?);
    built.insert(robot_source, kernels.clone());
    Ok(kernels)
}

fn build_kernels(device: &wgpu::Device, info: &wgpu::AdapterInfo, robot_source: &str) -> Result<Kernels> {
    // Constants and shared structs are generated from their Rust definitions.
    let source = [
        "alias Vec4 = vec4<f32>;\n",
        &format!("const CUBOID: u32 = {CUBOID}u;\nconst SPHERE: u32 = {SPHERE}u;\n"),
        &format!("const CYLINDER: u32 = {CYLINDER}u;\nconst CAPSULE: u32 = {CAPSULE}u;\nconst SDF: u32 = {SDF}u;\n"),
        &format!(
            "const MAX_HISTORY: u32 = {MAX_HISTORY}u;\nconst LINE_STEPS: u32 = {}u;\n\
             fn line_search(c: u32) -> f32 {{\n    var steps = array<f32, {}>({});\n    return steps[c];\n}}\n",
            LINE_SEARCH.len(),
            LINE_SEARCH.len(),
            LINE_SEARCH.map(|a| format!("{a:?}")).join(", ")
        ),
        GpuParams::WGSL,
        GpuLink::WGSL,
        GpuSphere::WGSL,
        GpuObstacle::WGSL,
        robot_source,
        include_str!("kernels.wgsl"),
    ]
    .concat();
    // Turn shader and pipeline validation failures into errors instead of panics.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batchplan kernels"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let buffer_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let mut entries = vec![buffer_entry(0, wgpu::BufferBindingType::Uniform)];
    let (ro, rw) = (1..=READ_ONLY_STORAGE, READ_ONLY_STORAGE + 1..=READ_ONLY_STORAGE + READ_WRITE_STORAGE);
    entries.extend(ro.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: true })));
    entries.extend(rw.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: false })));
    let layout0 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("main"), entries: &entries });
    let pl_main = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout0)],
        immediate_size: 0,
    });
    let pipeline = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl_main),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let evaluate = pipeline("evaluate_main");
    let ik = pipeline("ik_main");
    let traj_costs = pipeline("traj_costs");
    let traj_search = pipeline("traj_search");
    let traj_samples = pipeline("traj_samples");
    let traj_grad = pipeline("traj_grad");
    let lbfgs_direction = pipeline("lbfgs_direction");
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(Error::Gpu(format!("{} ({:?}) cannot build the kernels: {e}", info.name, info.backend)));
    }

    Ok(Kernels { layout0, evaluate, ik, traj_costs, traj_search, traj_samples, traj_grad, lbfgs_direction })
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
                ([], Some(name)) => Error::Gpu(format!("no Vulkan, Metal or DX12 adapter named like '{name}'")),
                ([], None) => Error::Gpu("no Vulkan, Metal or DX12 adapter found".into()),
                _ => Error::Gpu(format!("no GPU adapter can run the kernels:\n  {}", rejected.join("\n  "))),
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
        .map_err(|e| Error::Gpu(format!("requesting GPU device: {e}")))?;

        let built = Arc::new(Mutex::new(HashMap::new()));
        let kernels = kernels_for(&device, &info, &built, robot)?;
        let buffers = RobotBuffers::new(&device, robot);
        Ok(Self { robot: robot.clone(), info, device, queue, kernels, built, buffers })
    }

    fn params(&self, n_items: usize, w: &CollisionWeights) -> GpuParams {
        GpuParams {
            n_dof: self.robot.dof() as u32,
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
        if bytes > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) {
            return Err(Error::Gpu(format!(
                "the distance grids take {} MiB, more than {} can bind",
                bytes >> 20,
                self.info.name
            )));
        }
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
            &self.buffers.links,
            &self.buffers.spheres,
            &self.buffers.pairs,
            &self.buffers.limits,
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
            layout: &self.kernels.layout0,
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

    /// Blocks until the GPU has finished all submitted work.
    fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| Error::Gpu(e.to_string()))
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
        self.wait()?;
        rx.recv().map_err(|e| Error::Gpu(e.to_string()))?.map_err(|e| Error::Gpu(e.to_string()))?;
        let mapped = staging.slice(..).get_mapped_range().map_err(|e| Error::Gpu(e.to_string()))?;
        let out = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
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
            pass.set_pipeline(&self.kernels.evaluate);
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

    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>> {
        let kernels = kernels_for(&self.device, &self.info, &self.built, robot)?;
        let buffers = RobotBuffers::new(&self.device, robot);
        Ok(Box::new(Self { robot: robot.clone(), kernels, buffers, ..self.clone() }))
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
        let n = self.robot.dof();
        let mut errors = Vec::with_capacity(item_world.len());
        for ((chunk_world, chunk_targets), chunk_q) in
            item_world.chunks(IK_CHUNK).zip(targets.chunks(IK_CHUNK)).zip(q.chunks_mut(IK_CHUNK * n))
        {
            errors.extend(self.ik_chunk(worlds, chunk_world, chunk_targets, chunk_q, o)?);
        }
        Ok(errors)
    }

    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let worlds: &GpuWorlds = worlds.prepared();
        let (points, n) = (paths.points, paths.dof);
        if item_world.is_empty() || o.iterations == 0 {
            return Ok(());
        }
        // Paths are optimized in chunks whose L-BFGS state fits one binding and whose work per
        // submission stays well under driver watchdogs.
        let limits = self.device.limits();
        let bytes = 4 * lbfgs_stride(points, n, o) as u64;
        let fits = (limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) / bytes).max(1) as usize;
        let chunk = fits.min(TRAJ_CHUNK);
        for (chunk_world, chunk_paths) in item_world.chunks(chunk).zip(paths.positions.chunks_mut(chunk * points * n)) {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            self.trajopt_chunk(worlds, chunk_world, chunk_paths, points, o, deadline)?;
        }
        Ok(())
    }
}

/// Floats of L-BFGS state per path; must match `lbfgs_stride` in kernels.wgsl.
fn lbfgs_stride(points: usize, n: usize, o: &PlanOptions) -> usize {
    let samples = (points - 3) * o.samples_per_span;
    samples * n + LINE_SEARCH.len() * samples + (4 + 2 * o.history) * points * n + 5
}

impl GpuBackend {
    fn ik_chunk(
        &self,
        worlds: &GpuWorlds,
        item_world: &[u32],
        targets: &[Pose],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
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
                pass.set_pipeline(&self.kernels.ik);
                pass.set_bind_group(0, &bg, &[]);
                dispatch(pass, items);
            });
            done += params.iterations;
        }
        if o.iterations == 0 {
            params.iterations = 0;
            self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            self.submit_pass(|pass| {
                pass.set_pipeline(&self.kernels.ik);
                pass.set_bind_group(0, &bg, &[]);
                dispatch(pass, items);
            });
        }
        q.copy_from_slice(&self.read(&q_buf, q.len())?);
        Ok(self.read(&out, items * 2)?.chunks(2).map(|e| [e[0], e[1]]).collect())
    }

    fn trajopt_chunk(
        &self,
        worlds: &GpuWorlds,
        item_world: &[u32],
        traj: &mut [f32],
        points: usize,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let items = item_world.len();
        let mut params = self.params(items, &o.collision);
        params.points = points as u32;
        params.samples = o.samples_per_span as u32;
        params.w_acc = o.w_acc;
        params.w_vel = o.w_vel;
        params.initial_step = o.initial_step;
        params.history = o.history as u32;
        let params_buf = self.uniform(&params);
        let world_buf = storage(&self.device, "item world", item_world);
        let dummy = storage::<f32>(&self.device, "unused", &[]);
        let q_buf = storage(&self.device, "trajectories", traj);
        let aux = storage_zeroed(&self.device, "L-BFGS state", items * lbfgs_stride(points, self.robot.dof(), o));
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
        let samples = items * (points - 3) * o.samples_per_span;
        // The first round only prices the seeds: their directions are still zero.
        let rounds = o.iterations + 1;
        let mut k = 0;
        while k < rounds {
            let end = (k + TRAJ_ITERS_PER_SUBMIT).min(rounds);
            self.submit_pass(|pass| {
                pass.set_bind_group(0, &bg, &[]);
                for round in k..end {
                    pass.set_pipeline(&self.kernels.traj_costs);
                    dispatch(pass, samples * LINE_SEARCH.len());
                    pass.set_pipeline(&self.kernels.traj_search);
                    dispatch(pass, items);
                    if round + 1 == rounds {
                        break;
                    }
                    pass.set_pipeline(&self.kernels.traj_samples);
                    dispatch(pass, samples);
                    pass.set_pipeline(&self.kernels.traj_grad);
                    dispatch(pass, items * (points - 6));
                    pass.set_pipeline(&self.kernels.lbfgs_direction);
                    dispatch(pass, items);
                }
            });
            k = end;
            // Waiting for the GPU is only needed to keep a deadline.
            if let Some(d) = deadline {
                self.wait()?;
                if Instant::now() >= d {
                    break;
                }
            }
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
