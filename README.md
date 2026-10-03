# batchplan

Batched, collision-aware motion generation that runs on any GPU [wgpu](https://wgpu.rs) supports. That means Vulkan on AMD, NVIDIA and Intel, plus Metal and DX12. A CPU implementation produces the same results.

This is an MVP concept. It solves thousands of IK and trajectory-optimization problems at once, each in its own collision world. It also turns the results into training demonstrations, including recoveries from perturbed states. It needs no CUDA and no ROCm. It has been verified on three vendors: on an AMD Radeon 8060S through Mesa's stock Vulkan driver, on Apple Silicon through Metal, and on an NVIDIA T4 through NVIDIA's Vulkan driver. Datasets can be exported to LeRobot v3.0 through an optional bridge.

## Which GPUs

The kernels use only core features: 32-bit floats, with no subgroups, atomics or extensions. Any adapter wgpu exposes through Vulkan, Metal or DX12 should work if it allows 11 storage buffers per shader stage. Desktop drivers allow far more. Browser WebGPU and some embedded GPUs don't.

`Device::gpu` skips adapters that fall short. If none qualifies, it returns an error naming each adapter and what it lacks. OpenGL adapters are not used.

| Adapter | Status |
|---|---|
| AMD Radeon 8060S, Mesa RADV (Vulkan), Ubuntu 26.04 | Verified: all tests pass |
| Apple M4 Pro (Metal), macOS | Verified: all tests pass |
| NVIDIA Tesla T4, driver 595.91 (Vulkan), Ubuntu 24.04 (AWS g4dn.xlarge) | Verified: all tests pass |
| Mesa llvmpipe (Vulkan on the CPU) | Verified: benchmark solves the same problems |
| Other NVIDIA cards and Jetson, Intel, DX12 on Windows | Untested |

## What it does

| Module | Interface | Behind it |
|---|---|---|
| `device` | `Device::gpu(&robot)`, `Device::cpu(&robot)`, `device.evaluate(..)` | Robot uploaded once. Batched FK, sphere collision cost, analytic gradient and clearances. WGSL kernels on the GPU, rayon on the CPU. |
| `ik` | `solve_ik(&device, &worlds, &problems, &IkOptions)` | Many seeds per target. Damped least squares, with the collision gradient projected into the Jacobian null space. Success = pose tolerance + collision-free. |
| `trajopt` | `plan(&device, &worlds, &problems, &PlanOptions)` | Many seeds per start/goal. Adam on collision + smoothness cost, then validation by dense interpolation. `best()` picks the shortest valid seed. |
| `timing` | `retime(&robot, path, &RetimeOptions)` | Minimum-jerk (bell-shaped velocity) timing within the robot's velocity limits and an acceleration limit. |
| `datagen` | `demonstrations(&device, &worlds, &goals, &DemoOptions)` | The full demonstration pipeline; see [Training data](#training-data). `recovery_problems(..)` exposes the recovery step on its own. |
| `lerobot` (feature `lerobot`) | `lerobot::export(root, &robot, &worlds, &demos, &ExportOptions)` | Writes demonstrations as a LeRobot v3.0 dataset. |
| `robot`, `world`, `types` | `Robot::from_config_file`, `World`/`Obstacle`, `Pose`, `JointPaths`, `JointTrajectory` | URDF + collision-sphere config. Box and sphere obstacles. Shared data types with documented row-major shapes. |

Design choices:
- **Batch-first.** Every call covers many seeds, problems and worlds. Problems reference worlds by index, so one call can span thousands of different scenes.
- **One handle, two implementations.** `Device` is the only way to run batched work. The CPU and GPU implementations sit behind a crate-private trait and are tested against each other through `Device`.
- **Separate algorithms over shared types.** There are no planner plugins and no runtime configuration. Algorithms are plain functions over a `Device` and the shared data types, and they own seeding, validation and selection. That way both devices see identical inputs.
- **Standalone.** No middleware.

## Quick start

```bash
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                 # fail instead of skipping GPU tests without a GPU
cargo run --release --example bench -- 512                   # GPU vs CPU throughput
cargo run --release --example datagen -- data/demo 512 20    # demonstrations as .npy arrays (512 worlds, 20 fps)
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20   # as a LeRobot dataset
```

Set `BENCH_LLVMPIPE=1` to also run the WGSL kernels on the CPU through Mesa's llvmpipe Vulkan driver. To develop on one machine and run on a GPU box, use `REMOTE=user@host scripts/sync.sh '<command>'`. It rsyncs the checkout to `~/batchplan` there and runs the command.

```rust
use batchplan::*;
let robot = Robot::from_config_file("assets/franka/panda.json")?;
let device = Device::gpu(&robot)?;
let ik = solve_ik(&device, &worlds, &ik_problems, &IkOptions::default())?;
let problems: Vec<PlanProblem> = (0..ik_problems.len())
    .filter_map(|p| ik.best(p).map(|goal| PlanProblem { world: p as u32, start: robot.default_q().to_vec(), goal: goal.to_vec() }))
    .collect();
let result = plan(&device, &worlds, &problems, &PlanOptions::default())?;
```

## Results

All numbers are from a Framework Desktop: Ryzen AI Max+ 395 (16 cores / 32 threads), Radeon 8060S, Ubuntu 26.04, Mesa 26.0.3 RADV, no ROCm.

`bench -- 512` uses 512 random worlds, each a table plus 2–6 boxes. Each world gets a top-down grasp target.
- **IK:** 32 seeds × 60 iterations per target.
- **Planning:** 8 seeds × 32 waypoints × 200 Adam iterations from the default pose.

| Device | IK | Planning | Solved | End-to-end |
|---|---|---|---|---|
| Radeon 8060S (Vulkan, RADV) | 0.062 s (263k seeds/s) | 1.13 s (3.4k seeds/s) | 484/484 IK goals planned | **406 problems/s** |
| CPU, 32 threads (rayon) | 0.257 s | 5.49 s | 483/484 | 84 problems/s |
| CPU via llvmpipe (same WGSL), 128 worlds | 0.134 s | 2.79 s | 123/123 | 42 problems/s |

The same benchmark on a MacBook with an Apple M4 Pro (20-core GPU, 14 CPU cores):

| Device | IK | Planning | Solved | End-to-end |
|---|---|---|---|---|
| Apple M4 Pro (Metal) | 0.080 s | 2.00 s | 484/484 | **233 problems/s** |
| CPU, 14 threads (rayon) | 0.242 s | 5.39 s | 484/484 | 86 problems/s |

The same benchmark on an AWS g4dn.xlarge (NVIDIA Tesla T4, 4 vCPUs):

| Device | IK | Planning | Solved | End-to-end |
|---|---|---|---|---|
| Tesla T4 (Vulkan) | 0.115 s | 2.20 s | 484/484 | **209 problems/s** |
| CPU, 4 threads (rayon) | 5.01 s | 115.8 s | 483/484 | 4 problems/s |

`datagen -- data/demo 512 20` produced 481 nominal and 932 recovery episodes from 512 worlds on the Framework, in about 3 s of planning. Peak joint speed reaches 0.99 of the limit, and recovery starts sit a median 0.35 rad off the nominal path.

## Verification

`BATCHPLAN_REQUIRE_GPU=1 cargo test --release` runs 14 tests and passes on the Framework (Radeon, Vulkan), the Mac (M4 Pro, Metal) and an NVIDIA T4 (Vulkan). `--features lerobot` adds 2 export tests:
- **FK:** URDF forward kinematics matches Franka's published DH parameters to 1e-5.
- **Gradients:** analytic collision gradients match finite differences.
- **CPU IK:** solves targets taken from collision-free configurations.
- **Retiming:** respects velocity limits and keeps the endpoints.
- **GPU vs CPU, 20,000 configurations:** clearances agree to 3e-7 m, and gradients agree to 3e-4 relative.
- **GPU vs CPU IK:** the two agree on all 2,048 seeds.
- **GPU plans under an independent check:** every GPU plan reported valid was re-checked on the CPU at 4× denser interpolation. None penetrates; the worst clearance is +0.1 mm.
- **Adapter errors:** adapters below the required limits are rejected with the reason, and an unknown adapter name is a clear error.

## Training data

`datagen::demonstrations` turns goal poses into training demonstrations, batched on the device:
1. **IK:** solve each goal pose with many seeds; keep the collision-free solution with the most clearance.
2. **Plan nominal reaches:** start from the default pose plus joint noise (a collision-free sample), and plan to the IK goal.
3. **Plan recoveries:** perturb each solved path partway along it (20–80% by default), keep the collision-free perturbed states, and replan from them to the same goal. These are the recovery examples that raw planner data lacks.
4. **Retime:** apply a minimum-jerk (bell-shaped velocity) profile with a randomized speed scale, sampled at a fixed `dt`.

Each `Demonstration` holds its origin (nominal, or recovery with its parent and phase), world, goal pose and timed trajectory. `examples/datagen.rs` writes demonstrations in one of two formats.

### Plain arrays (default)

| File | Contents |
|---|---|
| `positions.npy`, `velocities.npy` | `[episodes, steps, dof]` float32. Padded past `length` with the final position and zero velocity. |
| `length.npy` | `[episodes]` int32, valid steps per episode |
| `kind.npy` | `[episodes]` uint8: 0 = nominal, 1 = recovery |
| `parent.npy` | `[episodes]` int32: for recoveries, the row of the nominal episode they branch from; -1 otherwise |
| `world.npy`, `worlds.json` | world index per episode and the obstacles of every world |
| `goal_pose.npy` | `[episodes, 7]` target of the `ee_link` frame: xyz + quaternion xyzw |
| `meta.json` | `dt`, joint names, velocity and acceleration limits, nominal/recovery counts, device |

### LeRobot v3.0 (optional bridge)

Build with `--features lerobot`, which adds the Arrow/Parquet dependencies. The core library doesn't depend on them. `lerobot::export` writes a dataset that `lerobot.datasets.LeRobotDataset(repo_id, root=path)` loads directly. The layout is LeRobot's standard v3.0 set: frame data, episode metadata and tasks as parquet, plus `info.json` and normalization `stats.json`. Each demonstration is one episode:

| Feature | Contents |
|---|---|
| `observation.state` | joint positions |
| `action` | joint positions of the next frame (absolute targets); the last frame repeats its own |
| `observation.environment_state` | goal pose (xyz, quaternion xyzw with w ≥ 0), then each obstacle as `[present, is_sphere, center xyz, half extents xyz, quaternion xyzw]`, zero-padded to the largest world |
| `is_recovery`, `parent_episode_index`, `world_index` | extensions; LeRobot policies only read `observation.*` and `action`, so these are ignored in training |
| `task` | "Move the gripper to the target pose." (`ExportOptions::task`) |

`meta/batchplan.json` adds the worlds, the environment-state layout and each episode's origin. There are no camera features: the data is state-only until rendering is added.

`scripts/validate_lerobot.py` checks an export with the real `lerobot` package (0.6.1). On a 712-episode export, every check passed:
- **Loading:** episodes, frames and the task string load as written.
- **Episodes:** boundaries are correct, and `action` is the next frame's state.
- **Labels and stats:** recovery labels match `meta/batchplan.json`, and the normalization stats match the data.
- **Action chunks:** chunks are padded correctly at episode ends.

`lerobot-train --policy.type=act --dataset.root=<export>` trains LeRobot's stock ACT policy on it from state plus environment state. A 50-step CPU run cut the loss from 45.6 to 5.9.

## Limits of the MVP

- **Geometry.** The robot is modeled as spheres only, and obstacles as boxes and spheres. There are no meshes, point clouds or depth-derived distance fields yet.
- **Kinematics.** The tree is limited to 16 actuated joints, 32 links and 128 spheres. Mimic joints must be locked.
- **Optimizer.** Trajectory optimization uses fixed-step Adam with no line search. Validation follows the linear interpolation between waypoints. Retiming does not bound accelerations at waypoint corners.
- **Kernel performance.** Kernels run one invocation per configuration (or per waypoint), with no shared memory or subgroup work. Buffers are allocated per call. This leaves performance on the table.
- **Training data.** Demonstrations are state-only reaches with the gripper held open, and every episode shares one task string. The LeRobot export writes all episode metadata to a single file, which caps it at roughly 100k episodes.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. The bundled Franka assets under `assets/franka` are Apache-2.0 (see below).

## Attribution

`assets/franka/franka_panda.urdf` is from franka_ros (Apache-2.0; see `assets/franka/LICENSE`). The collision spheres, self-collision buffers, ignore pairs and default pose in `assets/franka/panda.json` are converted from NVlabs/curobo `franka.yml` (Apache-2.0, © NVIDIA).
