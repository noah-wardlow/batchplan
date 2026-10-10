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
| `device` | `Device::gpu(&robot)`, `Device::cpu(&robot)`, `device.upload(&worlds)`, `device.evaluate(..)` | Robot uploaded once; worlds uploaded once into a `Worlds` handle that every algorithm takes. Batched FK, sphere collision cost, analytic gradient and clearances. WGSL kernels on the GPU, rayon on the CPU. Every batch is checked first (array shapes, world indices, worlds uploaded to this device), so malformed input is an `Err` on both devices. |
| `ik` | `solve_ik(&device, &worlds, &problems, &IkOptions)` | Many seeds per target. Damped least squares, with the collision gradient projected into the Jacobian null space. Success = pose tolerance + collision-free. The result keeps its problems; `ik.solved()` yields each one with its best configuration. |
| `trajopt` | `plan(&device, &worlds, &problems, &PlanOptions)` | Many seeds per start/goal, each a uniform cubic B-spline that starts and ends at rest. L-BFGS on smoothness plus collision cost sampled along the curve, with a line search that prices four step sizes at once, then validation by sampling the curve densely. Problems no seed solves fall back to RRT-Connect. `result.solved()` yields each problem with its shortest valid path's control points. |
| `rrt`, `shortcut` | `rrt::connect(&device, &worlds, &problems, &RrtOptions)`, `shortcut::shortcut(..)` | RRT-Connect with goal sets, and random shortcutting followed by redundant-waypoint removal; see [Fallback](#fallback-rrt-connect). |
| `timing` | `Trajectory::new(&robot, path, speed_scale)`, `trajectory.at(t, &mut state)`, `.sample(hz)`, `.check(&robot)` | Executable trajectories: timing that keeps position, velocity, acceleration and jerk within the robot's limits everywhere; allocation-free sampling for control loops; a check a safety layer can run. See [Executable trajectories](#executable-trajectories). |
| `datagen` | `demonstrations(&device, &worlds, &goals, &DemoOptions)` | The full demonstration pipeline; see [Training data](#training-data). `recovery_problems(&device, &worlds, &plan_result, ..)` exposes the recovery step on its own. |
| `npy` | `npy::export(root, &robot, &worlds, &demos, &ExportOptions)` | Writes demonstrations as plain `.npy` arrays. |
| `lerobot` (feature `lerobot`) | `lerobot::export(root, &robot, &worlds, &demos, &ExportOptions)` | Writes demonstrations as a LeRobot v3.0 dataset. |
| `robot`, `spheres` | `Robot::load(path, &RobotOptions)`, `CollisionModel::{load, save}` | Robots from URDF, MJCF or OpenUSD (feature `usd`), with mimic joints. Collision spheres are fitted to the links' geometry, or loaded from a committed collision-model file. See [Robots](#robots). |
| `world`, `sdf`, `types` | `World::load(path, &SdfOptions)`, `World`/`Obstacle`, `SdfGrid::{from_mesh, from_points, from_depth}`, `Pose`, `JointPaths`, `JointTrajectory`, `Solved` | Box, sphere, cylinder and capsule obstacles, and signed distance grids for anything else; static geometry from MJCF or USD scenes. See [Distance grids](#distance-grids). Shared data types with documented row-major shapes. |

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
- Per-joint kernel state must stay in arrays of `MAX_JOINTS` entries. Indexing it per link (32 entries) pushed it out of registers and cost 20% of GPU planning time on the Radeon.
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

- **Kinematics.** Revolute, continuous, prismatic and fixed joints. Mimic joints follow their leader (value = multiplier × leader + offset); the leader's range and velocity limit shrink so that every mimic joint stays within its own limits. Joints can be locked at a value. Up to 16 actuated joints, 16 moving joints (actuated plus mimic), 32 links and 128 spheres.
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

Test robots live in `assets/` with their licences ([assets/README.md](assets/README.md)):
- URDF: the Franka Panda (with cuRobo's hand-tuned spheres), the UR5e, the SO-101 and the Robotiq 2F-85 (one actuated joint driving five mimic joints);
- MJCF: MuJoCo Menagerie's Panda, UR5e and 2F-85;
- USD: newton-assets' UR5e and 2F-85, a Franka converted from our URDF with NVIDIA's urdf-usd-converter, and handwritten fixtures for each UsdPhysics convention.

## Distance grids

An `Obstacle::Sdf` places a signed distance grid in a world: distances on a regular grid in its own frame, stored as half floats and interpolated trilinearly on both devices. Worlds can share a grid; a device stores it once, and exports write it once (`worlds.json` holds `{"grids", "worlds"}`, with each `sdf` obstacle naming its grid by index). Three builders make grids, laid out by `SdfOptions` (1 cm voxels and 15 cm padding by default):
- **`SdfGrid::from_mesh`:** exact signed distances of a closed triangle mesh, with inside and outside decided by parry3d's pseudo-normals. Open or non-manifold meshes have no inside and are refused with the reason.
- **`SdfGrid::from_points`:** voxels holding a point are solid. A closed surface's inside reads as free beyond its shell.
- **`SdfGrid::from_depth`:** one depth image, its pinhole intrinsics and the camera pose. Each voxel is classified by projecting its center into the image, so an 8×8 time-of-flight array gives solid surfaces, not 64 points; dense images also mark the voxel of every pixel. The rule for unobserved space is explicit: space outside the image and along pixels without a reading is free, and space behind observed surfaces is free or solid as `Occlusion` says.

Grids err toward collision. Each grid point holds at most the true signed distance (for points and depth images, the distance to the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear interpolation can overestimate by. So a built grid never reads farther from an obstacle than its geometry, and a wall thinner than a voxel still blocks. Without the offset, interpolating exact samples across a 2 mm wall reads several millimeters of clearance inside it; a test pins both. The price is growth: up to a voxel diagonal for meshes, and about three voxels for points and depth images.

Outside its box, a grid reads the value at the nearest box point plus the distance to it. That is exact for collision checking when the box reaches at least the collision margin plus the robot's largest sphere radius beyond every surface, which is what `SdfOptions::padding` is for.

`examples/depth.rs` renders a 640×480 depth image of a tabletop scene, builds a grid from it (172×192×72 points in 33 ms on the M4 Pro), and plans Panda reaches with the grid as the only obstacle. IK reaches 48 of 64 grasp targets; the others lie in the camera's shadow, which `Occlusion::Occupied` treats as solid. All 48 plan, and their closest approach to the true scene is 34 mm. The arm is not in the rendered image; a real camera's pixels on the robot must be masked out first.

## Executable trajectories

A planned path is a uniform cubic B-spline over its control points. The first three and last three equal the start and goal, so the path starts and ends at rest.
- **Optimization and validation see the same curve.** Trajectory optimization samples collision cost along the spline (`samples_per_span` points per span), and validation samples it densely (`validate_substeps` per span).
- **Timing bounds the whole curve, not samples of it.** On a B-spline with knot interval `h`:
  - velocity is a quadratic B-spline over the control-point differences divided by `h`;
  - acceleration is a linear one over the second differences divided by `h²`;
  - jerk is constant per span: third differences divided by `h³`.

  Each stays within its largest control value. `Trajectory::new` picks the smallest `h` that keeps every joint within its velocity, acceleration and jerk limits, then divides by the speed scale. Limits come from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default).
- **For control loops.** `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating. A test with a counting allocator holds it to that. `sample(hz)` returns fixed-rate samples for datasets.
- **For safety layers.** A `Trajectory` is plain serializable data, so a planner process can hand it to a controller process. `trajectory.check(&robot)` refuses one that is non-finite, not at rest at both ends, out of a joint range, or over any velocity, acceleration or jerk limit anywhere along its length.

## Fallback: RRT-Connect

Trajectory optimization cannot escape a seed that has to go around an obstacle the long way. Problems where no seed validates fall back to sampling-based planning (`PlanOptions::fallback`, on by default):
1. **RRT-Connect** (Kuffner and LaValle, 2000) grows a tree from the start and one from the goal set. Problems advance in lockstep: each round extends every unsolved problem's tree toward 8 random configurations and connects its other tree toward each new node. Each of those two steps checks every problem's edges, sampled every 0.02 rad, in one batched `evaluate`, so a GPU sees few large batches rather than many tiny ones.
2. **Shortcutting** tries 8 random shortcuts per path per round and keeps the one that shortens it most. It stops after 32 failures in a row, then drops waypoints whose neighbors see each other. On the UR5e test paths the result is within 5% of the straight-line lower bound.
3. **Tracing** turns the waypoints into B-spline control points: each waypoint three times, the rest along the edges. The spline then runs exactly along the checked path and stops at its corners. Spreading control points evenly by arc length instead cut corners into the obstacles RRT-Connect had found its way around. The traced spline and an optimized version are validated like any seed, and the shorter valid one is the result.

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

`BATCHPLAN_REQUIRE_GPU=1 cargo test --release` runs 59 tests; `--features lerobot` adds 3 export tests and `--features usd` adds 6 OpenUSD tests. Both configurations pass on the Framework (Radeon, Vulkan) and the Mac (M4 Pro, Metal). An earlier version of the suite (14 tests at commit `950e06c`) also passed on an NVIDIA T4 (Vulkan).
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
- **Exporters:** the `.npy` export round-trips trajectories, padding and labels, and the LeRobot export writes consistent v3.0 metadata. Both refuse to overwrite an existing dataset.
- **Obstacle distances:** every obstacle kind returns unit gradients that match finite differences, stepping back along the gradient lands on the surface, and cylinder and capsule distances match closed forms. The GPU agrees with the CPU for each kind.
- **Distance grids:**
  - Gradients match finite differences inside the grid and beyond its box, away from cell faces; so do collision gradients through a grid.
  - The GPU agrees with the CPU to 3e-7 m (Metal and RADV), including two grids in one world, one of them shared with another world and stored after a grid with an odd number of points.
  - Mesh grids never read farther than the true distance and at most a voxel diagonal nearer. Point grids never read beyond their solid voxels. Open and non-manifold meshes are refused.
  - Depth images from an 8×8 time-of-flight array and from a 640×480 camera both reproduce the table under them, follow both occlusion rules, and leave unobserved space free.
  - MJCF scene meshes collide as their convex hulls; USD ones as `physics:approximation` says. Exports write each grid once, and LeRobot's environment state describes a grid by its box.
  - Planting a wrong grid offset or half-float order in the kernel, a wrong gradient axis, a missing conservative offset, a depth builder that only marks pixel points, or ignored hull semantics fails a test.
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

## Training data

`datagen::demonstrations` turns goal poses into training demonstrations, batched on the device:
1. **IK:** solve each goal pose with many seeds; keep the collision-free solution with the most clearance.
2. **Plan nominal reaches:** start from the default pose plus joint noise (a collision-free sample), and plan to the IK goal.
3. **Plan recoveries:** perturb each solved path partway along it (20–80% by default), keep the collision-free perturbed states, and replan from them to the same goal. These are the recovery examples that raw planner data lacks.
4. **Time:** each trajectory runs at a random fraction (60–100% by default) of the fastest timing within the robot's limits, sampled at a fixed `dt`.

Each `Demonstration` holds its origin (nominal, or recovery with its parent and phase), world, goal pose and timed trajectory. Two exporters write them, `npy::export` and `lerobot::export`. `examples/datagen.rs` uses one or the other. Generated from the same worlds, the two formats hold identical trajectories, labels, worlds and goals.

### Plain arrays (`npy::export`, default)

| File | Contents |
|---|---|
| `positions.npy`, `velocities.npy` | `[episodes, steps, dof]` float32. Padded past `length` with the final position and zero velocity. |
| `length.npy` | `[episodes]` int32, valid steps per episode |
| `kind.npy` | `[episodes]` uint8: 0 = nominal, 1 = recovery |
| `parent.npy` | `[episodes]` int32: for recoveries, the row of the nominal episode they branch from; -1 otherwise |
| `world.npy`, `worlds.json` | world index per episode, and `{"grids", "worlds"}`: the obstacles of every world, with each distance grid written once |
| `goal_pose.npy` | `[episodes, 7]` target of the `ee_link` frame: xyz + quaternion xyzw |
| `meta.json` | `dt`, joint names, velocity, acceleration and jerk limits, nominal/recovery counts, device |

### LeRobot v3.0 (optional bridge)

Build with `--features lerobot`, which adds the Arrow/Parquet dependencies. The core library doesn't depend on them. `lerobot::export` writes a dataset that `lerobot.datasets.LeRobotDataset(repo_id, root=path)` loads directly. The layout is LeRobot's standard v3.0 set: frame data, episode metadata and tasks as parquet, plus `info.json` and normalization `stats.json`. Each demonstration is one episode:

| Feature | Contents |
|---|---|
| `observation.state` | joint positions |
| `action` | joint positions of the next frame (absolute targets); the last frame repeats its own |
| `observation.environment_state` | goal pose (xyz, quaternion xyzw with w ≥ 0), then each obstacle as `[present, kind, center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world. `kind` is 0 cuboid, 1 sphere, 2 cylinder, 3 capsule, 4 distance grid; `size` is the half extents of a cuboid or of a grid's box, else (radius, radius, half height or half length) |
| `is_recovery`, `parent_episode_index`, `world_index` | extensions; LeRobot policies only read `observation.*` and `action`, so these are ignored in training |
| `task` | "Move the gripper to the target pose." (`ExportOptions::task`) |

`meta/batchplan.json` adds the worlds, the environment-state layout and each episode's origin. There are no camera features: the data is state-only until rendering is added.

`scripts/validate_lerobot.py` checks an export with the real `lerobot` package (0.6.1). On the 512-world export (1,401 episodes, 65,186 frames), every check passed:
- **Loading:** episodes, frames and the task string load as written.
- **Episodes:** boundaries are correct, and `action` is the next frame's state.
- **Labels and stats:** recovery labels match `meta/batchplan.json`, and the normalization stats match the data.
- **Action chunks:** chunks are padded correctly at episode ends.

`lerobot-train --policy.type=act --dataset.root=<export>` trains LeRobot's stock ACT policy on it from state plus environment state. On the same export, a 50-step CPU run cut the loss from 44.2 (step 10) to 5.8.

## Limits of the MVP

- **Geometry.** The robot is modeled as spheres only. Distance grids are built on the CPU, from one depth image at a time (no fusion over frames), and depth images must have the robot masked out by the caller.
- **Kinematics.** The tree is limited to 16 actuated joints, 16 moving joints, 32 links and 128 spheres. Floating, planar and ball joints aren't supported; closed loops (MJCF `connect`, USD loop joints) are dropped from the tree.
- **Long motions.** Joint ranges are intervals, not circles: continuous joints are planned within ±π and nothing wraps. On the UR5e, a goal whose shoulder has turned past the table below is unreachable even for RRT-Connect.
- **Timing.** Trajectories are rest-to-rest and not time-optimal: bounding the B-spline by its control points is conservative. They cannot start from a moving state.
- **Kernel performance.** Kernels run one invocation per configuration (or per collision sample, or per control point), with no shared memory or subgroup work. Worlds stay on the device, but per-call buffers (configurations, paths) are allocated per call. This leaves performance on the table.
- **Training data.** Demonstrations are state-only reaches with the gripper held open, and every episode shares one task string. The LeRobot export writes all episode metadata to a single file, which caps it at roughly 100k episodes.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. The test robots under `assets/` keep their own licences (see below).

## Attribution

[assets/README.md](assets/README.md) lists each test robot's source, licence and changes. In short: the Franka Panda is from franka_ros and NVlabs/curobo (Apache-2.0); its collision spheres in `assets/franka/panda_collision.json` are converted from cuRobo's `franka.yml` (Apache-2.0, © NVIDIA). The UR5e is from Universal Robots' ROS 2 description (BSD-3-Clause), the Robotiq 2F-85 from PickNik's ros2_robotiq_gripper (BSD-3-Clause), and the SO-101 from TheRobotStudio's SO-ARM100 (Apache-2.0).
