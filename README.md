# batchplan

Batched, collision-aware motion generation that runs on any GPU [wgpu](https://wgpu.rs) supports. That means Vulkan on AMD, NVIDIA and Intel, plus Metal and DX12. A CPU implementation produces the same results.

This is an MVP concept. It solves thousands of IK and trajectory-optimization problems at once, each in its own collision world. It also turns the results into training demonstrations, including recoveries from perturbed states. It needs no CUDA and no ROCm. It has been verified on three vendors: on an AMD Radeon 8060S through Mesa's stock Vulkan driver, on Apple Silicon through Metal, and on an NVIDIA T4 through NVIDIA's Vulkan driver. Datasets can be exported to LeRobot v3.0 through an optional bridge.

## Which GPUs

The kernels use only core features: 32-bit floats, with no subgroups, atomics or extensions. Any adapter wgpu exposes through Vulkan, Metal or DX12 should work if it allows 12 storage buffers per shader stage. Desktop drivers allow far more. Browser WebGPU and some embedded GPUs don't.

`Device::gpu` skips adapters that fall short. If none qualifies, it returns an error naming each adapter and what it lacks. OpenGL adapters are not used.

| Adapter | Status |
|---|---|
| AMD Radeon 8060S, Mesa RADV (Vulkan), Ubuntu 26.04 | Verified: all tests pass |
| Apple M4 Pro (Metal), macOS | Verified: all tests pass |
| NVIDIA Tesla T4, driver 595.91 (Vulkan), Ubuntu 24.04 (AWS g4dn.xlarge) | Verified at commit `950e06c` (the MVP's 14 tests); not re-run since |
| Mesa llvmpipe (Vulkan on the CPU) | Verified: benchmark solves the same problems |
| Other NVIDIA cards and Jetson, Intel, DX12 on Windows | Untested |

## What it does

| Module | Interface | Behind it |
|---|---|---|
| `device` | `Device::gpu(&robot)`, `Device::cpu(&robot)`, `Device::cpu_threads(&robot, n)`, `device.with_robot(&robot)`, `device.upload(&worlds)`, `device.evaluate(..)` | Robot uploaded once; worlds uploaded once into a `Worlds` handle that every algorithm takes. The CPU device runs on threads it owns. Batched FK, sphere collision cost, analytic gradient and clearances. WGSL kernels on the GPU, rayon on the CPU. Every batch is checked first (array shapes, world indices, worlds uploaded to this device), so malformed input is an `Err` on both devices. |
| `ik` | `solve_ik(&device, &worlds, &problems, &IkOptions)` | Many seeds per target. Damped least squares, with the collision gradient projected into the Jacobian null space. Success = pose tolerance + collision-free. The result keeps its problems; `ik.solved()` yields each one with its best configuration. |
| `trajopt` | `plan(&device, &worlds, &problems, &PlanOptions)` | Many seeds per start/goal, each a uniform cubic B-spline that starts and ends at rest. L-BFGS on smoothness plus collision cost sampled along the curve, with a line search that prices four step sizes at once, then validation by sampling the curve densely. Problems no seed solves fall back to RRT-Connect. `result.solved()` yields each problem with its shortest valid path's control points. |
| `rrt`, `shortcut` | `rrt::connect(&device, &worlds, &problems, &RrtOptions)`, `shortcut::shortcut(..)` | RRT-Connect with goal sets, and random shortcutting followed by redundant-waypoint removal; see [Fallback](#fallback-rrt-connect). |
| `timing` | `Trajectory::new(&robot, path, speed_scale)`, `trajectory.at(t, &mut state)`, `.sample(hz)`, `.check(&robot)` | Executable trajectories: timing that keeps position, velocity, acceleration and jerk within the robot's limits everywhere; allocation-free sampling for control loops; a check a safety layer can run. See [Executable trajectories](#executable-trajectories). |
| `datagen` | `demonstrations(&device, &worlds, &goals, &DemoOptions)` | The full demonstration pipeline; see [Training data](#training-data). `recovery_problems(&device, &worlds, &plan_result, ..)` exposes the recovery step on its own. |
| `npy` | `npy::export(root, &robot, &worlds, &demos, &ExportOptions)` | Writes demonstrations as plain `.npy` arrays. |
| `lerobot` (feature `lerobot`) | `lerobot::export(root, &robot, &worlds, &demos, &ExportOptions)` | Writes demonstrations as a LeRobot v3.0 dataset. |
| `robot`, `spheres` | `Robot::load(path, &RobotOptions)`, `robot.attach(&AttachedObject)`, `CollisionModel::{load, save}` | Robots from URDF, MJCF or OpenUSD (feature `usd`), with mimic joints. Collision spheres are fitted to the links' geometry, or loaded from a committed collision-model file. See [Robots](#robots). |
| `render` | `Camera { name, width, height, intrinsics, mount }`, `device.render(&worlds, &camera, &item_world, &q, &moved)` | Colour and depth images of worlds and the robot from fixed or link-mounted cameras, batched on the device. See [Camera images](#camera-images). |
| `meshes` | `MeshModel::new(&robot)`, `meshes.clearance(&worlds, &item_world, &q)` | Clearances measured on the links' meshes rather than their spheres, on the CPU, for checking finished trajectories. See [Checking against meshes](#checking-against-meshes). |
| `world`, `sdf`, `types` | `World::load(path, &SdfOptions)`, `World`/`Obstacle`, `SdfGrid::{from_mesh, from_points, from_depth}`, `OccupancyMap`, `Pose`, `JointPaths`, `JointTrajectory`, `Solved` | Box, sphere, cylinder and capsule obstacles, and signed distance grids for anything else; static geometry from MJCF or USD scenes. See [Distance grids](#distance-grids). Shared data types with documented row-major shapes. |

Design choices:
- **Batch-first.** Every call covers many seeds, problems and worlds. Problems reference worlds by index, so one call can span thousands of different scenes.
- **One handle, two implementations.** `Device` is the only way to run batched work. The CPU and GPU implementations sit behind a crate-private trait and are tested against each other through `Device`.
- **Separate algorithms over shared types.** There are no planner plugins and no runtime configuration. Algorithms are plain functions over a `Device` and the shared data types, and they own seeding, validation and selection. That way both devices see identical inputs.
- **Shared structs declared once.** Every struct passed to the GPU (parameters, links, spheres, obstacles) and every shared constant (kernel limits, line-search steps) is declared once in Rust, and its WGSL declaration is generated from it. The two sides can't drift apart.
- **Standalone.** No middleware.

## Quick start

```bash
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                 # fail instead of skipping GPU tests without a GPU
cargo run --release --example bench -- 512                   # GPU vs CPU throughput
cargo run --release --example datagen -- data/demo 512 20    # demonstrations as .npy arrays (512 worlds, 20 fps)
cargo run --release --example depth                         # plan around what a depth camera sees
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20   # as a LeRobot dataset
```

Set `BENCH_LLVMPIPE=1` to also run the WGSL kernels on the CPU through Mesa's llvmpipe Vulkan driver. To develop on one machine and run on a GPU box, use `REMOTE=user@host scripts/sync.sh '<command>'`. It rsyncs the checkout to `~/batchplan` there and runs the command.

```rust
use batchplan::*;
let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
let device = Device::gpu(&robot)?;
let worlds = device.upload(&scenes)?; // obstacles and distance grids stay on the device
let ik = solve_ik(&device, &worlds, &ik_problems, &IkOptions::default())?;
// Each solved IK problem carries its world, so the handoff can't mix up worlds.
let problems: Vec<PlanProblem> = ik
    .solved()
    .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec() })
    .collect();
let result = plan(&device, &worlds, &problems, &PlanOptions::default())?;
for s in result.solved() {
    let trajectory = Trajectory::new(&robot, s.solution, 1.0); // as fast as the robot's limits allow
    let mut state = JointState::new(robot.dof());
    trajectory.at(0.5 * trajectory.duration(), &mut state); // position, velocity, acceleration
}
```

## Results

`bench -- 512` uses 512 random worlds, each a table plus 2–6 boxes, with one top-down grasp target per world:
- **IK:** 32 seeds × 60 iterations per target.
- **Planning:** 8 seeds × 24 control points × 40 L-BFGS iterations from the default pose, with collision checked at 3 points per span (63 per seed, priced at 4 step sizes per iteration).

Every device solves the same 484 of 512 IK targets and plans all 484.

| Machine | GPU | GPU end-to-end | CPU end-to-end | GPU speedup |
|---|---|---|---|---|
| Framework Desktop (Ryzen AI Max+ 395) | AMD Radeon 8060S, Mesa RADV (Vulkan) | **190 problems/s** | 37 problems/s (32 threads) | 5.1× |
| MacBook (Apple M4 Pro) | Apple M4 Pro, 20 cores (Metal) | **150 problems/s** | 36 problems/s (14 threads) | 4.2× |

GPU time per phase on the Radeon: 0.064 s for IK (16,384 seeds) and 2.5 s for planning (3,872 seeds). Each L-BFGS iteration prices four step sizes, so these 40 iterations do about as much collision checking as 200 of Adam's. On the M4 Pro they run 35% faster than Adam did; on the Radeon 13% slower. The M4 Pro was measured while the machine ran other work. Checking collision along the curve instead of at 30 waypoints roughly doubled planning time compared with the MVP, which reached 406 problems/s on the Radeon and 209 on an NVIDIA T4 (commit `950e06c`). The benchmark below shows what that buys.

Performance claims come from alternating runs of two builds on one machine. Two things those runs caught:
- Hot functions the CPU backend calls across modules are marked `#[inline]`. Without it, the CPU path lost 15–20% whenever unrelated code changed how the compiler split the crate.
- Collision kernels must not index private arrays at run time. Mesa's RADV moves such arrays into scratch memory once they exceed 256 bytes, and per-link and per-sphere arrays cost 3–5 KB of it per invocation. Writing the kernels out for each robot (frames and collision wrenches as separate variables) made GPU planning 4.8× faster on the Radeon and 2.7× faster on the M4 Pro.
- The line search's cost passes must skip the collision gradient. Computing it there, unused, made GPU planning 28% slower on both GPUs.
- RRT-Connect must extend toward several configurations per round. One per round meant thousands of tiny GPU calls for problems it cannot solve, which halved benchmark throughput.

`datagen -- --lerobot data/lerobot_512 512 20` produced 481 nominal and 920 recovery episodes from 512 worlds on the M4 Pro, in about 9 s of planning.

## Robots

`Robot::load` picks a loader by file extension. Each loader only translates its format into one crate-private description; kinematics, spheres and self-collision analysis never see the format.
- **URDF.** `package://` paths resolve through `RobotOptions::package_dirs` and then through the directories above the file.
- **MJCF (`.xml`, `.mjcf`).** A native parser covering what kinematics and collision need: `<include>`, `<default>` classes and `childclass`, the compiler's angle units (degrees unless told otherwise), mesh directory and Euler sequence, every orientation form, bodies with several joints, `fromto`, and `<equality><joint>` as mimic joints. Body subtrees with a joint are the robot; jointless bodies and world geoms are the scene. `connect`/`weld` loops cannot be represented in a tree and are ignored. MJCF has no velocity limits, so `RobotOptions::max_velocity` (2 rad/s) applies.
- **OpenUSD (`.usd`, `.usda`, `.usdc`, `.usdz`; feature `usd`).** Built on the pure-Rust `openusd` crates (0.7.0, pinned).
  - **Links and joints:** UsdPhysics rigid bodies are links. The tree grows from the world over Revolute, Prismatic and Fixed joints, using `localPos`/`localRot` on both sides. Joints authored from child to parent are flipped; loop-closing and excluded joints are skipped.
  - **Units:** stage units (`metersPerUnit`, centimeters when unset) and Y-up stages are converted. Revolute limits and angular velocity limits are degrees.
  - **Mimic joints:** from `NewtonMimicAPI` or `PhysxMimicJointAPI`.
  - **Collision geometry:** from `PhysicsCollisionAPI` prims, including instanced ones, with unauthored schema sizes and transform scale applied.
  - **Frames:** ghost-link `Xform`s (such as an end-effector frame) become fixed frames.
  - **Variants:** `RobotOptions::variants` selects variants where the file authors no selection.
- **Scenes.** `World::load` reads the static geometry of an MJCF or USD scene as obstacles; planes become slabs and ellipsoids their bounding boxes. Meshes become [distance grids](#distance-grids) of their convex hulls where the format collides them that way: MJCF always (as MuJoCo does), USD when `physics:approximation` is `convexHull`. Other USD approximations use the exact mesh.

- **Kinematics.** Revolute, continuous, prismatic and fixed joints. Planar, ball and floating joints (URDF `planar`/`floating`, MJCF `free`/`ball`, USD spherical joints) become chains of one-axis joints: `<joint>_x`, `_y`, `_z` for translations, whose limits come from `RobotOptions::joint_limits`, and `_theta` or `_rx`, `_ry`, `_rz` (intrinsic x-y-z angles, singular where `_ry` reaches ±90°) for rotations. Continuous joints turn without limit: planning takes the short way round, interpolation and distances wrap (RRT-Connect included), and no position limit applies. For any joint whose range spans more than a turn (the UR5e's ±2π), a goal is reached at whichever equivalent, a whole turn away, is nearest or free: trajectory optimization seeds toward several, and RRT-Connect takes them all as goals. Closed loops (MJCF `connect`, USD joints that close a loop) are solved when loading: each loop's passive joints follow its actuated joint at their static equilibrium, the least spring energy that closes the loop within their limits (what MuJoCo settles to without contact), fitted by quartics. Menagerie's Robotiq 2F-85 loads as one joint whose pads track MuJoCo's settled positions within 0.2 mm. Mimic joints follow their leader by a polynomial of degree up to four (URDF and USD mimics are linear, MJCF joint equalities quartic); the leader's range and velocity limit shrink so that every mimic joint stays within its own limits. Joints can be locked at a value. Robots have no fixed size limit (a 40-joint, 41-link, 130-sphere test robot runs on both devices).
- **Collision spheres.** Without a collision model, spheres are fitted to each link's collision geometry (or its visual geometry) with cuRobo's voxel method:
  1. Interior grid points become candidate spheres that touch the surface.
  2. A greedy cover picks the candidates that reach the most surface samples.
  3. One overhang is searched for across the whole robot, the smallest at which the cover fits the budget (64 spheres by default).
  4. Spheres grow until 20,000 surface samples per link are inside them, so the model is conservative.

  The model's `source` records how far its spheres reach beyond the geometry. With 64 spheres that is 3.0 cm on the UR5e and the Panda, 1.9 cm on the SO-101 and 1.2 cm on the 2F-85. Sharp convex edges cost the most spheres. Fresh surface samples land at most 0.8 mm outside the UR5e's spheres.
- **Self-collision pairs** follow MoveIt's setup assistant:
  - adjacent links are skipped;
  - so are pairs whose spheres already overlap at the default configuration;
  - so are pairs whose geometry never touches across 10,000 random configurations. This test uses the meshes, not the spheres, so sphere overhang does not keep pairs that cannot really collide.

  `RobotOptions::srdf` adds a MoveIt SRDF's disabled pairs.
- **Collision-model files.** `robot.collision_model().save(path)` writes the spheres, self-collision buffers and ignored pairs as JSON. Commit the file, tune it by hand if needed, and load it back through `RobotOptions::collision_model`.
- **Mounted robots.** Links that no joint moves (the base and anything fixed to it) are not checked against the world: they touch it the same way in every configuration, so a robot standing on its table is not in collision with it. They still collide with the robot's moving links.

Test robots live in `assets/` with their licences ([assets/README.md](assets/README.md)):
- URDF: the Franka Panda (with cuRobo's hand-tuned spheres), the UR5e, the SO-101 and the Robotiq 2F-85 (one actuated joint driving five mimic joints);
- MJCF: MuJoCo Menagerie's Panda, UR5e and 2F-85;
- USD: newton-assets' UR5e and 2F-85, a Franka converted from our URDF with NVIDIA's urdf-usd-converter, and handwritten fixtures for each UsdPhysics convention.

### Checking against meshes

Planning and validation run on spheres, which only approximate the links. `MeshModel::new(&robot)` loads each link's collision meshes (or visual meshes, where spheres were fitted to those), and `clearance(&worlds, &item_world, &q)` measures world and self clearance between meshes for a batch of configurations, as MoveIt checks a planning scene with FCL. It runs on the CPU, for finished trajectories rather than inner loops:
- **Distances are exact.** They are parry3d's, searched by branch and bound: pairs in order of their bounding spheres' gap, then each pair's convex hulls, then the meshes. That makes it about 110 µs per Panda configuration on the M4 Pro's 14 threads, down from 620 µs.
- **Meshes are surfaces, as in FCL.** A link entirely inside another closed mesh does not touch it.
- **Distance grids hold no exact surface.** Against them, a link's clearance is the grid's distance at points covering its surface to within 5 mm, less 5 mm.

`cargo run --release --example benchmark -- --meshes` checks every successful plan the same way, at 4× the planner's validation density:

| Spheres | Ends free | IK + plan (spheres) | Also clear on meshes |
|---|---|---|---|
| cuRobo's hand-tuned (the default) | 98.1% | 95.6% | 88.5% |
| Fitted to the meshes (`--fitted`) | 40.0% | 49.7% | 49.7% |

cuRobo's spheres leave parts of the Panda's collision meshes uncovered, by up to 18 mm on `panda_link5`, so about one plan in thirteen grazes an obstacle on the meshes. Spheres fitted to the meshes cover them: every plan clears them, but their 3 cm overhang already puts most of these tight scenes' start or goal states in collision. Checking plans with a `MeshModel` keeps the fast spheres and drops the rare grazing plan.

## Distance grids

An `Obstacle::Sdf` places a signed distance grid in a world: distances on a regular grid in its own frame, stored as half floats and interpolated trilinearly on both devices. Worlds can share a grid; a device stores it once, and exports write it once (`worlds.json` holds `{"grids", "worlds"}`, with each `sdf` obstacle naming its grid by index). Three builders make grids, laid out by `SdfOptions` (1 cm voxels and 15 cm padding by default), and an occupancy map fuses depth frames into them. Point clouds, depth images and maps become grids on the `Device` they are given (see [building grids on the GPU](#building-grids-on-the-gpu)):
- **`SdfGrid::from_mesh`:** exact signed distances of a closed triangle mesh, with inside and outside decided by parry3d's pseudo-normals. Open or non-manifold meshes have no inside and are refused with the reason.
- **`SdfGrid::from_points(&device, ..)`:** voxels holding a point are solid. A closed surface's inside reads as free beyond its shell.
- **`SdfGrid::from_depth(&device, ..)`:** one `DepthImage` (readings, pinhole intrinsics and the camera pose). Each voxel is classified by projecting its center into the image, so an 8×8 time-of-flight array gives solid surfaces, not 64 points; dense images also mark the voxel of every pixel. The rule for unobserved space is explicit: space outside the image and along pixels without a reading is free, and space behind observed surfaces is free or solid as `Occlusion` says.
- **Masking the robot.** A camera that sees the arm would make it an obstacle to itself. `DepthImage::robot_depth(&robot, &q, padding)` gives each pixel's depth to the robot's collision spheres at `q`, grown by `padding` (ray-sphere hits, searched only in each sphere's pixel box). Passed to `from_depth` or `OccupancyMap::integrate`, readings on the robot are not obstacles: space in front of the arm is free, space behind it follows `Occlusion`, and anything between the camera and the arm stays.
- **`OccupancyMap`:** depth frames fused over time on a fixed world grid, as OctoMap fuses them. Each voxel holds the log-odds that it is occupied, one byte in twentieths: a hit adds 0.85, a miss subtracts 0.4, clamped to [−2, 3.5]. Something seen once and then seen through three times is free again; voxels behind readings, outside the image or beyond the map are not updated. `integrate(&device, ..)` fuses a frame and `grid(&device, unknown)` builds a distance grid, with voxels never observed solid or free as `unknown` says.

Grids err toward collision. Each grid point holds at most the true signed distance (for points and depth images, the distance to the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear interpolation can overestimate by. So a built grid never reads farther from an obstacle than its geometry, and a wall thinner than a voxel still blocks. Without the offset, interpolating exact samples across a 2 mm wall reads several millimeters of clearance inside it; a test pins both. The price is growth: up to a voxel diagonal for meshes, and about three voxels for points and depth images.

Outside its box, a grid reads the value at the nearest box point plus the distance to it. That is exact for collision checking when the box reaches at least the collision margin plus the robot's largest sphere radius beyond every surface, which is what `SdfOptions::padding` is for.

`examples/depth.rs` renders a 640×480 depth image of a tabletop scene, builds a grid of 172×192×72 points from it, and plans Panda reaches with the grid as the only obstacle. IK reaches 48 of 64 grasp targets; the others lie in the camera's shadow, which `Occlusion::Occupied` treats as solid. All 48 plan, and their closest approach to the true scene is 34 mm. The arm is not in the rendered image; with a real camera, pass `robot_depth` to mask it out.

### Building grids on the GPU

On a GPU device, every step after the host computes a depth image's bounds runs in compute kernels (`src/grids.wgsl`):
1. classifying each voxel against the image;
2. marking the voxels that hold readings;
3. a map's log-odds update;
4. the distance transform;
5. finishing each value as a half float.

The CPU device runs the same steps on its threads.
- **The transform is exact on both devices.** It is Felzenszwalb and Huttenlocher's separable transform in integers, one invocation per line on the GPU, so GPUs and CPUs build identical grids from the same occupancy.
- **Finishing uses a table.** Each stored value depends only on a voxel's squared distance and whether it is occupied. The host builds that table up to the grid's squared diagonal while the GPU runs the transform; this is why a grid's diagonal is limited to 4,096 voxels.
- **Classification can differ slightly.** It projects voxel centres in floating point, so a voxel within rounding of a pixel's edge or a reading's depth may land either way. In a 640×480 image with the arm masked out, that is 2 of 600,000 voxels on Metal and 51 on RADV.
- **Buffers are reused.** The GPU keeps its grid buffers for the next build; wgpu zero-fills new buffers, which cost more than the transform.

Median build times over 21 runs, against the previous commit's CPU-only builders, for a 640×480 image of a table seen from 1 m:

| | Before (CPU) | CPU | GPU |
|---|---|---|---|
| Radeon 8060S / Ryzen AI Max+ 395, 1 cm grid (0.96M points) | 21.1–22.7 ms | 8.8–9.5 ms | 1.9–2.0 ms |
| Radeon 8060S, 5 mm grid (7.5M points) | 116–121 ms | 47–49 ms | 10.4–10.9 ms |
| Radeon 8060S, fusing one frame into a 1.8M-point map | 6.8–7.0 ms | 4.6–5.0 ms | 0.53–0.64 ms |
| Radeon 8060S, that map's grid | 23.1–23.9 ms | 6.1–6.3 ms | 2.4–2.6 ms |
| M4 Pro, 1 cm grid | 11.6–12.1 ms | 6.4–6.5 ms | 3.4–3.7 ms |
| M4 Pro, 5 mm grid | 74–75 ms | 36 ms | 14.6–15.0 ms |
| M4 Pro, fusing one frame | 3.1–3.2 ms | 2.2–2.3 ms | 1.45–1.6 ms |
| M4 Pro, the map's grid | 16.7–16.9 ms | 6.8–7.0 ms | 4.4–4.5 ms |

The CPU gains come from the integer transform (it was in `f64`, with an allocation per column), parallel finishing, and parallel back-projection. Masking the robot (`robot_depth`) went from 2.5 ms on one thread to 0.5 ms on the M4 Pro.

## Executable trajectories

A planned path is a uniform cubic B-spline over its control points. The first three and last three equal the start and goal, so the path starts and ends at rest.
- **Optimization and validation see the same curve.** Trajectory optimization samples collision cost along the spline (`samples_per_span` points per span), and validation samples it densely (`validate_substeps` per span).
- **Timing is time-optimal along the planned path, without changing it.** A trajectory is the path `P(s)` and a time map `s = σ(t)`, itself a uniform cubic B-spline. `Trajectory::new` computes two timings and keeps the faster:
  - **Time-optimal** (TOPP-RA, Pham & Pham 2018):
    - the fastest timing of the path under joint velocity and acceleration limits, plus a cap from jerk's speed term (the path's third derivative times ṡ³);
    - smoothed into a B-spline time map (Schoenberg's approximation of s(t));
    - where the smoothed map overshoots, the speed caps there are lowered and the profile solved again, three rounds; a final stretch takes out whatever overshoot remains when sampled at 64 points per span.

    On planned Panda paths it is 12% faster than uniform timing on average and never slower, at about 0.5 ms per trajectory.
  - **Uniform**: `σ(t) = t / h`. Velocity is then a quadratic B-spline over the control-point differences divided by `h`, acceleration a linear one over second differences divided by `h²`, and jerk constant per span (third differences over `h³`). The smallest `h` that keeps every joint within its limits bounds the whole curve exactly.

  Either is then slowed by the speed scale. Limits come from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default).
- **From a moving start.** `PlanProblem::start_motion` gives the robot's current velocity and acceleration, for replanning mid-motion. The path's first three control points then continue that motion: `q0 − v0 h0 + a0 h0²/3`, `q0 − a0 h0²/6` and `q0 + v0 h0 + a0 h0²/3`, with `h0` a quarter slower than a straight path's uniform timing. `Trajectory::moving` times such a path from exactly that state, starting the time-optimal profile at ṡ = 1/h0 with no s̈. Neither timing stretches a moving start, since that would change it, so a path that cannot brake in time is refused (`Error::Unsafe`). `check_from(&robot, &state)` verifies that a trajectory starts at a given state.
- **For control loops.** `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating. A test with a counting allocator holds it to that. `sample(hz)` returns fixed-rate samples for datasets.
- **For safety layers.** A `Trajectory` is plain serializable data, so a planner process can hand it to a controller process. `trajectory.check(&robot)` refuses one that is non-finite, not at rest at both ends, out of a joint range, running backward, or over a velocity, acceleration or jerk limit: anywhere along its length for uniform timing, at 64 points per time-map span otherwise (time-optimal maps are stretched 0.3% beyond what those points need, which covers the curve between them).

## Fallback: RRT-Connect

Trajectory optimization cannot escape a seed that has to go around an obstacle the long way. Problems where no seed validates fall back to sampling-based planning (`PlanOptions::fallback`, on by default):
1. **RRT-Connect** (Kuffner and LaValle, 2000) grows a tree from the start and one from the goal set. Problems advance in lockstep: each round extends every unsolved problem's tree toward 8 random configurations and connects its other tree toward each new node. Each of those two steps checks every problem's edges, sampled every 0.02 rad, in one batched `evaluate`, so a GPU sees few large batches rather than many tiny ones.
2. **Shortcutting** tries 8 random shortcuts per path per round and keeps the one that shortens it most. It stops after 32 failures in a row, then drops waypoints whose neighbors see each other. On the UR5e test paths the result is within 5% of the straight-line lower bound.
3. **Tracing** turns the waypoints into B-spline control points: each waypoint three times, the rest along the edges. The spline then runs exactly along the checked path and stops at its corners. Spreading control points evenly by arc length instead cut corners into the obstacles RRT-Connect had found its way around. The traced spline and an optimized version are validated like any seed, and the shorter valid one is the result.

## Embedding in a control process

batchplan can run inside a robot's own controller process, next to a fixed-rate control loop:
- **CPU-only builds.** `default-features = false` drops the `gpu` feature and wgpu with it; CI builds that for `aarch64-unknown-linux-gnu`. `Device::gpu` then returns an error instead of a device.
- **Threads the host controls.** `Device::cpu_threads(&robot, n)` starts `n` workers and runs every batch on them, never on rayon's global pool; a test counts the process's threads around IK, planning, RRT-Connect and shortcutting with a one-thread device. Building distance grids classifies and transforms voxels on the device's threads. Loading a robot that needs sphere fitting, mesh grids, and the host's share of grid building (a depth image's bounds, the finishing table, packing occupancy for a GPU) use rayon's global pool; call them inside `pool.install(..)` of the host's own rayon pool to keep them there.
- **Time budgets.** `PlanOptions::time_budget` stops optimization and the fallback between rounds (on a GPU, between submissions of 8 rounds) and returns the best valid paths so far. On the M4 Pro, a 1 ms budget for 64 problems returns in 25 ms on the CPU (3.0 s without a budget) and in 104 ms on the GPU (507 ms), where one submission takes about 110 ms. Without a budget, every run gives bit-identical results.
- **Batch-1 latency.** When a batch has fewer paths than the CPU device has threads, each path's line search and gradient spread over threads too. On the Framework's CPU, IK and planning for one problem at a time went from 117–131 / 142–161 / 268–370 ms (mean / p75 / p98, two runs) to 63–65 / 78–79 / 140–149 ms, with batched throughput unchanged.
- **Typed errors.** Every public function returns a `batchplan::Error` whose kind a host can act on: `Input` (shapes, indices, options, geometry), `Load` (with the file's path), `Gpu`, `Threads`, `Unsafe` (`Trajectory::check`) and `Write`.
- **Holding objects.** `robot.attach(&AttachedObject)` follows MoveIt: the object becomes a frame fixed to a link, with spheres fitted to its shapes; it collides with the world and the rest of the robot, never with its link or the `touch_links` that hold it. `detach` takes it off. `device.with_robot(&held)` gives a device for the new robot that shares the old one's GPU or threads and its uploaded worlds, so picking something up costs no new device and no new upload.

`examples/control_loop.rs` puts these together, shaped like a ros2_control or robotd controller. A planner thread with a two-thread CPU device plans reaches under a 500 ms budget, checks each trajectory, and hands it over through a latest-value slot. The 50 Hz loop takes a new trajectory when one is ready without ever waiting on the lock, samples it with the allocation-free `at(t)`, and holds its pose otherwise. On the M4 Pro, while other work loaded the machine, plans took 150–180 ms and the loop's worst tick was 5 ms late.

## Benchmark

`scripts/fetch_benchmark.sh` downloads the standard Panda problem sets that [robometrics](https://github.com/fishbotics/robometrics) packages as plain YAML (MIT): MotionBenchMaker's 800 problems in 8 sets and MπNets' 1,800 in 12, at a pinned commit with checksums. No ROS is involved. `cargo run --release --example benchmark` runs every set on the GPU and the CPU in two modes:
- **plan:** plan from the start to the set's first IK solution (planning only);
- **ik+plan:** solve IK for the goal pose, then plan to the best solution.

A problem succeeds when the final `panda_hand` position is within 1 cm of the goal, every joint is within its limits, and an independent CPU check at 4× the planner's validation density finds no collision. That check uses the same sphere model the planner uses. "Free ends" counts the problems whose start and given IK goal are collision-free under that model, which caps the achievable success. Throughput runs each set as one batch; batch-1 latency runs IK and planning for one problem at a time (`--latency N`: the first N problems of each set).

All 2,600 problems on the Framework Desktop, at three stages: straight-line paths retimed (M0), B-spline trajectories optimized by Adam (M3), and L-BFGS with the RRT-Connect fallback (M5, now). Latency covers the first 20 problems of each set for M0 and the first 50 since.

| Device | Free ends | Plan | IK + plan | Batched | Batch-1 latency (mean / p75 / p98) |
|---|---|---|---|---|---|
| Radeon 8060S (Vulkan), M0 | 98.1% | 81.2% | 74.2% | 196 problems/s | 162 / 197 / 503 ms |
| Radeon 8060S (Vulkan), M3 | 98.1% | 92.2% | 87.5% | 157 problems/s | 104 / 111 / 201 ms |
| Radeon 8060S (Vulkan), now | 98.1% | **97.2%** | **95.4%** | 134 problems/s | 85 / 90 / 193 ms |
| Ryzen AI Max+ 395 CPU (32 threads), M0 | 98.1% | 81.3% | 74.2% | 31 problems/s | 88 / 104 / 251 ms |
| Ryzen AI Max+ 395 CPU (32 threads), M3 | 98.1% | 92.2% | 87.5% | 18 problems/s | 106 / 127 / 242 ms |
| Ryzen AI Max+ 395 CPU (32 threads), now | 98.1% | **97.3%** | **95.0%** | 16 problems/s | 121 / 156 / 272 ms |

The M4 Pro's GPU and CPU solve the same problems.
- **Where the gains come from:** L-BFGS alone reaches 94.8% / 90.5% on the Radeon at 163 problems/s, with latency 77 / 82 / 127 ms: more problems solved, faster, than Adam. The RRT-Connect fallback adds the rest for 14% of throughput.
- **The hardest sets gain most** (M0 → M3 → now):
  - `table_under_pick`: 23% → 62% → 96%;
  - `cubby_task_oriented`: 43% → 87% → 97% planning only;
  - `dresser_task_oriented`: 63% → 86% → 97% planning only.
- **Motion quality** (finite differences at 100 Hz, medians over sets):
  - Peak acceleration fell from 20 rad/s² (M0) to 4.9 rad/s², within the 5 rad/s² limit.
  - Peak jerk fell from about 2,000 rad/s³ to 46 rad/s³, within the 50 rad/s³ limit.
  - Motion time fell from 2.55 s to 2.30 s with L-BFGS.

GPU latency at batch size 1 is dominated by per-call setup and readback, which is why the CPU is competitive there.

Uploading worlds once and adding distance grids changed no success rate. In alternating runs against the previous commit on the Framework Desktop, `bench -- 512` planned 4% faster on the GPU (1,775 against 1,708 seeds/s) and 8% faster on the CPU (320 against 296); the benchmark's throughput and latency stayed within run-to-run noise. On the M4 Pro, nothing changed measurably.

## Verification

`BATCHPLAN_REQUIRE_GPU=1 cargo test --release` runs 98 tests; `--features lerobot` adds 6 export tests and `--features usd` adds 7 OpenUSD tests. Without default features (CPU only), 90 tests run. All configurations pass on the Framework (Radeon, Vulkan) and the Mac (M4 Pro, Metal), and CI runs them on Linux with the kernels on Mesa's llvmpipe. An earlier version of the suite (14 tests at commit `950e06c`) also passed on an NVIDIA T4 (Vulkan).
- **FK:** URDF forward kinematics matches Franka's published DH parameters to 1e-5.
- **Collision gradients:** analytic gradients match finite differences.
- **Trajectory optimization:**
  - The first L-BFGS step has no history, so it is steepest descent. Its direction, scaled to its largest entry, matches finite differences of the smoothness cost and of the full cost on every device.
  - GPU and CPU step directions agree to 3e-5 at the first step and to 2e-4 at the fourth, which comes from the two-loop recursion over three remembered steps. Both devices lower the full trajectory cost of 240 paths alike: from 13,161 to 972 and 974 on the M4 Pro, from 13,949 to 961 and 962 on the Radeon.
  - The two-loop recursion equals an explicitly built BFGS inverse Hessian, and a line search that finds nothing cheaper resets the history.
  - Planting a swapped weight in the GPU parameter packing, steepest ascent, an ignored step size, a reversed recursion loop, a stuck history ring or a wrong scaling fails a test.
- **GPU vs CPU, 20,000 configurations:** clearances agree to 3e-7 m, and gradients agree to 3e-4 relative.
- **GPU vs CPU IK:** the two agree on all 2,048 seeds.
- **GPU plans under an independent check:** every GPU plan reported valid was re-checked on the CPU at 4× denser interpolation. None penetrates; the worst clearance is +0.1 mm.
- **Every device behaves the same:**
  - Malformed batches (wrong array lengths, a world index past the end, worlds uploaded to another device) are an `Err` on both CPU and GPU.
  - Obstacle-free worlds plan on both.
  - Plans keep the worlds of their problems when one world holds several targets.
- **CPU IK and retiming:** IK solves targets taken from collision-free configurations; retiming respects velocity limits and keeps the endpoints.
- **RRT-Connect and shortcutting:**
  - Every edge RRT-Connect returns is collision-free under a CPU check four times denser, on both devices, and paths start at the start and end at a goal.
  - Shortcutting never lengthens a path or moves its ends, and every new edge passes the same denser check.
  - Fixed seeds give identical paths; changing either seed changes them.
  - With the fallback, `plan` solves every UR5e problem RRT-Connect can reach, and problems trajectory optimization solved keep their paths.
  - Planting coarse edge checks, trees that never join, an unreversed path, a lengthening shortcut, an ignored seed, an arc-length refit or no fallback fails a test.
- **Exporters:** the `.npy` export round-trips trajectories, padding, labels, grippers, carried poses and tasks, and the LeRobot export writes consistent v3.0 metadata. A 1,500-episode export with 1 MB files spreads frames and episode metadata over several files, each episode row naming its own file and the file holding its frames, and its statistics match the frames. Tasks are indexed per frame, the gripper is the last state dimension, and a carried box moves in the environment state frame by frame. Both refuse to overwrite an existing dataset. Planting rows that all claim the first metadata file, episodes weighted equally in the statistics, a carried box left in place, one task index, or an unpadded gripper fails a test.
- **Pick-and-place:** on each device, 6 of 6 tabletop tasks succeed. The gripper closes once and moves gradually, the box is set down within 3 mm of the place pose, held at the grasp point without slipping, closed on across its narrower side, the arm never touches the scene, the lifted box clears it, and velocities agree with positions. Seeded IK stays on the branch it starts from. Planting a grasp across the wider side, unreversed velocities, a box left behind, a grasp above the box, a gripper that never reopens, lowering to the wrong pose, or a seed that IK ignores fails a test.
- **Obstacle distances:** every obstacle kind returns unit gradients that match finite differences, stepping back along the gradient lands on the surface, and cylinder and capsule distances match closed forms. The GPU agrees with the CPU for each kind.
- **Distance grids:**
  - Gradients match finite differences inside the grid and beyond its box, away from cell faces; so do collision gradients through a grid.
  - The GPU agrees with the CPU to 3e-7 m (Metal and RADV), including two grids in one world, one of them shared with another world and stored after a grid with an odd number of points.
  - Mesh grids never read farther than the true distance and at most a voxel diagonal nearer. Point grids never read beyond their solid voxels. Open and non-manifold meshes are refused.
  - On every device: depth images from an 8×8 time-of-flight array and from a 640×480 camera both reproduce the table under them, follow both occlusion rules, and leave unobserved space free. A fused map keeps the table, holds a ball seen through twice and clears it the third time, leaves space never observed to the rule given, and takes no hits from readings beyond its edge. Masked out of a camera's image, the Panda's spheres read free in grids and maps alike while a box in front of the arm stays solid.
  - The GPU builds exactly the CPU's grids from point clouds and maps, and classifies a 640×480 image's voxels and fuses it into a map with at most 0.05% of voxels apart. The CPU's integer transform reproduces the earlier floating-point one bit for bit.
  - MJCF scene meshes collide as their convex hulls; USD ones as `physics:approximation` says. Exports write each grid once, and LeRobot's environment state describes a grid by its box.
  - Planting a wrong grid offset or half-float order in the kernel, a wrong gradient axis, a missing conservative offset, a depth builder that only marks pixel points, ignored hull semantics, a map that never clears, readings clamped onto a map's edge, or a mask that ignores the arm or drops what is in front of it fails a test. So do a wrongly rounded parabola intersection or a late switch between parabolas in either device's transform, and, in the GPU's kernels, hits or classification that ignore the robot, misses that never clear, a half-pixel projection error, surfaces thinner than a voxel, free hidden space, and a log-odds byte written to the wrong place.
- **Trajectories:**
  - Planned trajectories stay within velocity, acceleration and jerk limits along their whole length, reach the binding limit, and start and end exactly at rest.
  - The analytic velocity and acceleration match finite differences.
  - `at` makes no allocations. `check` refuses a NaN, running too fast, a jerk spike, a moving start, leaving a joint range and the wrong joint count.
  - The trajectory gradient (smoothness plus collision sampled along the spline) matches finite differences of the full cost on every device. Planting a wrong basis weight in either device's gather, a wrong jerk exponent in the timing, or an allocation in `at` fails a test.
- **Robots from URDF:**
  - The 2F-85's link poses match an independent composition of its URDF with mimic joints applied. Its collision gradient matches finite differences, and the GPU agrees with the CPU. Planting a dropped multiplier in the CPU kinematics, the CPU gradient or the GPU gradient fails a test.
  - Mimic limits intersect into the leader's range and velocity.
  - Fitted spheres contain independently sampled link surfaces, and fitting is deterministic.
  - Collision-model files round-trip, SRDF pairs stop being checked, and `package://` paths resolve.
  - The UR5e, SO-101 and Panda (with fitted spheres) each plan 12 of 12 tabletop motions on the CPU and the GPU.
  - A Panda with fitted spheres, whose base spheres reach into the table it stands on, is collision-free on it and plans on every device, while a pin through its elbow still collides. Planting the base back into either device's world check fails the test.
- **Camera images:** on every device, a camera looking straight down sees each obstacle kind's top at its depth (within 0.1 mm; a grid within its slack), a lying cylinder and capsule apart from upright ones, and each top shaded as lit from above; a sphere's silhouette ends where projection says. The robot's depths match an independent ray caster within 0.5 mm with identical silhouettes, a wrist camera follows forward kinematics, and moved obstacles move. GPU and CPU differ in under 0.5% of pixels of a cluttered scene. Exported PNGs decode to exactly the rendered colours and millimetres, with a carried box where it is. Planting cylinder caps facing the wrong way, an ignored moved obstacle, capsule ends at the centre, flat grid normals, a dropped camera rotation, missing robot spheres, little-endian depth, unscaled colour statistics, unrendered carried boxes, or depth in metres fails a test.
- **Mesh clearances:**
  - Distances to every obstacle kind match closed forms to 0.1 mm, follow the joints, stay exact among several obstacles in any order, and read zero across a surface. Self clearance between links matches too, and a concave link is measured on its surface, not its hull.
  - Against a grid of a cube, clearance stays within the grid's documented slack. Held objects are checked as part of the robot, and the Panda's meshes load and read clear of its own table.
  - Fitted spheres never read farther than the meshes over 300 random configurations.
  - Planting cylinders along the wrong axis, unturned link poses, grid points left in the link frame, dropped self pairs, vertex-only grid checks, dropped held shapes, an unsorted search, a search that stops at the hulls, or hulls in place of meshes fails a test.
- **MJCF:**
  - Menagerie's Panda matches our URDF Panda link for link to 1e-5.
  - Menagerie's UR5e matches the UR5e URDF up to constant per-link frame offsets (within 3 mm; Menagerie rounds a few dimensions), and plans like the others.
  - A handwritten model pins MuJoCo's conventions against hand-composed transforms: degrees by default, intrinsic Euler sequences, `axisangle`, `xyaxes`, `zaxis`, joint anchors, several joints per body, includes, default classes and `fromto`. Planting radians as the default, extrinsic Euler composition, ignored anchors, or static bodies in the robot fails a test.
- **OpenUSD:**
  - Handwritten fixtures pin UsdPhysics conventions against hand-composed transforms:
    - Y-up centimeters, a payload and a variant fallback;
    - flat and nested bodies, `localPos1` ≠ 0, a joint authored child to parent;
    - both mimic schemas, an instanced collider, and unauthored primitive sizes.
  - Planting a wrong up-axis rotation, ignored units, an unflipped swapped joint, an ignored `localPos1`, mimic offsets left in degrees, a wrong PhysX gearing sign, skipped instance proxies or a wrong default cube size fails a test.
  - newton-assets' UR5e matches the Menagerie MJCF it was converted from exactly. Their 2F-85 mimics its driver and fits spheres to mesh colliders read from a binary layer.
  - The Franka converted from our URDF matches it to 1e-5, with identical self-collision results under the same collision model.
  - `scripts/validate_usd.py` rebuilds every fixture's kinematics with Pixar's `usd-core` and agrees with batchplan to within 6e-6.
- **Adapter errors:** adapters below the required limits are rejected with the reason, and an unknown adapter name is a clear error.
- **Embedding:**
  - A one-thread CPU device starts exactly one thread and no others while it plans; planting one use of rayon's global pool fails the test.
  - A 1 ms time budget returns within one more chunk of work than the budget, on the CPU and the GPU, and the fallback stops at the budget too. Without a budget, two runs give identical paths. Planting a backend or a fallback that ignores the deadline fails a test.
  - Errors have the documented kinds: a malformed batch is `Input`, an unknown adapter `Gpu`, a bad file `Load` with its path, an unsafe trajectory `Unsafe`.
  - An attached box moves with the hand, collides with an obstacle between the fingers that the bare hand misses, and collides with the fingers unless they are touch links. Collision gradients through it match finite differences. Detaching restores the robot exactly. Devices made with `with_robot` plan around the box in worlds uploaded to the original device, and see it exactly as a device made for the held robot does, on the CPU and the GPU. Planting ignored touch links, a wrong kinematic chain, a GPU device that keeps the old robot, or a detach that keeps the spheres fails a test.

## Training data

`datagen::demonstrations` turns goal poses into training demonstrations, batched on the device:
1. **IK:** solve each goal pose with many seeds; keep the collision-free solution with the most clearance.
2. **Plan nominal reaches:** start from the default pose plus joint noise (a collision-free sample), and plan to the IK goal.
3. **Plan recoveries:** perturb each solved path partway along it (20–80% by default), keep the collision-free perturbed states, and replan from them to the same goal. These are the recovery examples that raw planner data lacks.
4. **Time:** each trajectory runs at a random fraction (60–100% by default) of the fastest timing within the robot's limits, sampled at a fixed `dt`.

`datagen::pick_and_place` turns tasks (a box in a world and where to set it down) into pick-and-place demonstrations. Each episode chains segments planned in batches, all from rest to rest:
1. **Approach** a pose 10 cm above the box from the default configuration, with the box an obstacle.
2. **Descend** to a top-down grasp across the box's narrower side, the box no longer an obstacle as the fingers close around it, then **close** the gripper over half a second.
3. **Lift** back up and **transfer** to above the place pose holding the box (`Robot::attach`, touching only the links fixed to the wrist, on a `Device::with_robot` device that shares the uploaded worlds).
4. **Lower** it, **open** the gripper and **retreat**.

Short moves keep the arm on one IK branch: IK for the grasp and the place pose starts from the configuration above them (`IkProblem::seed`) and takes the solution nearest it (`IkResult::nearest`). Each episode records the gripper's opening and where the box is at every sample, and a task such as "Pick up the box and put it 20 cm to the left." `examples/datagen.rs --pick-place` completes 492 of 512 random tabletop tasks in 20 s on the M4 Pro's GPU.

Each `Demonstration` holds its origin (nominal, or recovery with its parent and phase), world, task, goal pose, timed trajectory, the gripper's opening per sample (1 open, 0 closed; reaches hold it open), and any obstacle it carries with its pose per sample. Two exporters write them, `npy::export` and `lerobot::export`. `examples/datagen.rs` uses one or the other. Generated from the same worlds, the two formats hold identical trajectories, labels, worlds and goals.

### Camera images

`device.render(&worlds, &camera, &item_world, &q, &moved)` draws a batch of views: the robot at each configuration in its world, with an obstacle moved to a pose per view where `moved` says so (a carried box). A `Camera` has a name, an image size, pinhole intrinsics (OpenCV axes) and a mount: fixed in the world (`Mount::World(pose)`) or on a robot link (`Mount::Link { link, offset }`, a wrist camera). Each pixel's ray meets the primitives exactly, distance grids by sphere tracing, and the robot as its collision spheres. It gets the depth along the camera's axis (0 where nothing is hit) and a colour: one per obstacle by its index in the world, grey for the robot, lit from above. The GPU (`render.wgsl`) and the CPU (`render.rs`) cast the same rays; a 128×128 view takes about 65 µs on either machine's GPU (15,500 views/s) and 165 µs on its CPU (6,000 views/s).

### Plain arrays (`npy::export`, default)

| File | Contents |
|---|---|
| `positions.npy`, `velocities.npy` | `[episodes, steps, dof]` float32. Padded past `length` with the final position and zero velocity. |
| `length.npy` | `[episodes]` int32, valid steps per episode |
| `kind.npy` | `[episodes]` uint8: 0 = nominal, 1 = recovery |
| `parent.npy` | `[episodes]` int32: for recoveries, the row of the nominal episode they branch from; -1 otherwise |
| `world.npy`, `worlds.json` | world index per episode, and `{"grids", "worlds"}`: the obstacles of every world, with each distance grid written once |
| `goal_pose.npy` | `[episodes, 7]` the pose the episode drives toward (the `ee_link` frame's for a reach, the box's for pick-and-place): xyz + quaternion xyzw |
| `gripper.npy` | `[episodes, steps]` float32, the gripper's opening (1 open, 0 closed), padded with its final value |
| `carried.npy`, `carried_pose.npy` | `[episodes]` int32, the obstacle the episode moves (-1 if none), and `[episodes, steps, 7]` float32, its pose at each step (zero if none) |
| `task.npy` | `[episodes]` int32, each episode's index into `meta.json`'s `tasks` |
| `meta.json` | `dt`, joint names, velocity, acceleration and jerk limits, nominal/recovery counts, tasks, device |

### LeRobot v3.0 (optional bridge)

Build with `--features lerobot`, which adds the Arrow/Parquet and PNG dependencies. The core library doesn't depend on them. `lerobot::export(root, &device, &worlds, &demos, &options)` writes a dataset that `lerobot.datasets.LeRobotDataset(repo_id, root=path)` loads directly. The layout is LeRobot's standard v3.0 set: frame data, episode metadata and tasks as parquet, plus `info.json` and normalization `stats.json`. Frame data and episode metadata are written as they are converted, each starting a new file past `ExportOptions::data_files_size_in_mb` (100 MB, LeRobot's default) and a new chunk every 1,000 files. Dataset statistics combine the episodes' as LeRobot's `aggregate_feature_stats` does, so nothing holds every frame and datasets can exceed memory. Each demonstration is one episode:

| Feature | Contents |
|---|---|
| `observation.state` | joint positions, then the gripper's opening |
| `action` | the next frame's state (absolute targets); the last frame repeats its own |
| `observation.environment_state` | goal pose (xyz, quaternion xyzw with w ≥ 0), then each obstacle where it is at that frame (so a carried box moves) as `[present, kind, center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world. `kind` is 0 cuboid, 1 sphere, 2 cylinder, 3 capsule, 4 distance grid; `size` is the half extents of a cuboid or of a grid's box, else (radius, radius, half height or half length) |
| `observation.images.<camera>` | for each of `ExportOptions::cameras`, the colour image as an inline PNG `image` feature `[height, width, 3]`, rendered on the device with the carried box where it is |
| `observation.images.<camera>_depth` | with `ExportOptions::depth_images`, the depth in millimetres as a 16-bit PNG `[height, width, 1]` flagged `is_depth_map`. Off by default: LeRobot 0.6's ACT and diffusion policies read every camera feature as colour |
| `is_recovery`, `parent_episode_index`, `world_index` | extensions; LeRobot policies only read `observation.*` and `action`, so these are ignored in training |
| `task` | the episode's task string, indexed in `meta/tasks.parquet` |

`meta/batchplan.json` adds the worlds, the environment-state layout, each episode's origin and the obstacle it carries. Image statistics are exact per channel (from histograms of every pixel), in [0, 1] for colour and millimetres for depth, nested `[channels][1][1]` as LeRobot nests them. `examples/datagen.rs --images [--depth]` records a front camera and a wrist camera beside the Panda's hand at 128×128.

`scripts/validate_lerobot.py` checks an export with the real `lerobot` package (0.6.1). Every check passed on four exports: 512 reach worlds (1,401 episodes, 65,186 frames), 2,048 reach worlds written with 1 MB files (5,693 episodes and 216,902 frames across 6 episode-metadata files), 512 pick-and-place tasks (492 episodes, 89,753 frames, 18 tasks), and 32 pick-and-place tasks with colour and depth images from two cameras:
- **Loading:** episodes, frames and task strings load as written, from every file. Colour images load as `(3, H, W)` tensors in [0, 1], depth as `(1, H, W)` in millimetres, with statistics shaped `(C, 1, 1)`.
- **Episodes:** boundaries are correct, and `action` is the next frame's state.
- **Labels and stats:** recovery labels match `meta/batchplan.json`, and the normalization stats match the data.
- **Action chunks:** chunks are padded correctly at episode ends.

`lerobot-train --policy.type=act --dataset.root=<export>` trains LeRobot's stock ACT policy on it from state plus environment state. A 50-step CPU run cut the loss from 44.2 (step 10) to 5.8 on the reach export, and from 45.4 to 5.8 on the pick-and-place one. With the two cameras' colour images, ACT's image backbone trains on them too (loss 46.7 to 15.6 in 20 steps).

## Limits of the MVP

- **Geometry.** Planning runs on spheres; `MeshModel` checks results against the meshes on the CPU, as a filter after planning.
- **Kinematics.** A closed loop must have exactly one actuated joint, and its passive joints must follow it by a quartic to within 0.5 mm of closure; other loops are rejected when loading. Ball and floating rotations are Euler angles, so they lose a direction of motion where the middle angle reaches ±90°.
- **Timing.** Time-optimal timing smooths a velocity- and acceleration-optimal profile, so it is not jerk-optimal. Trajectories end at rest.
- **Kernel performance.** Small batches stay latency-bound on the GPU: one IK-and-plan query takes about 16 ms on the Radeon and 29 ms on the M4 Pro, mostly serial IK iterations. Shader modules without bounds checks would add 6–12% but need `unsafe`.
- **Training data.** Images draw the robot as its collision spheres and the scene with flat colours, not photorealistic. Pick-and-place grasps upright boxes from above, one object size per held robot, and its episodes have no recoveries.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. The test robots under `assets/` keep their own licences (see below).

## Attribution

[assets/README.md](assets/README.md) lists each test robot's source, licence and changes. In short: the Franka Panda is from franka_ros and NVlabs/curobo (Apache-2.0); its collision spheres in `assets/franka/panda_collision.json` are converted from cuRobo's `franka.yml` (Apache-2.0, © NVIDIA). The UR5e is from Universal Robots' ROS 2 description (BSD-3-Clause), the Robotiq 2F-85 from PickNik's ros2_robotiq_gripper (BSD-3-Clause), and the SO-101 from TheRobotStudio's SO-ARM100 (Apache-2.0).
