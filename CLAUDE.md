# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

batchplan is a Rust library for batched, collision-aware robot motion generation: IK and trajectory optimization over thousands of seeds, problems and worlds at once. It runs on any GPU wgpu supports (Vulkan, Metal, DX12) without CUDA or ROCm, and has a CPU implementation that produces the same numbers. Its main purpose is generating robot-learning training data: nominal demonstrations plus *recoveries* from perturbed states, exported as `.npy` arrays or LeRobot v3.0 datasets.

It is an MVP. It has been verified on an AMD Radeon 8060S (Mesa RADV), an Apple M4 Pro (Metal) and an NVIDIA T4 (Vulkan).

Deliberate scope decisions:
- No Python bindings. Python appears only in validation scripts.
- Browser WebGPU is deferred. The kernels need 11 storage buffers per stage; browsers allow 8–10.
- The main open gap for VLA training is camera images: data is state-only.

## Commands

```bash
cargo build --release --all-targets [--features lerobot]
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                    # 37 tests; without the env var, GPU tests skip silently when no adapter exists
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features lerobot # + 2 export tests
cargo test --release --test gpu trajopt_gradients_match_cpu_element_wise   # one test (test files: cpu, gpu, device, export, robot, trajectory)
cargo fmt --check                                               # rustfmt.toml: max_width 120
cargo clippy --release --all-targets [--features lerobot]       # keep at zero warnings, both configurations
cargo doc --no-deps --features lerobot                          # keep at zero warnings
cargo run --release --example bench -- 512                      # GPU vs CPU throughput; BENCH_LLVMPIPE=1 adds the WGSL kernels on Mesa's CPU Vulkan driver
scripts/fetch_benchmark.sh && cargo run --release --example benchmark   # MotionBenchMaker + MπNets (2,600 Panda problems), GPU and CPU
cargo run --release --example datagen -- data/demo 512 20       # .npy dataset: <out_dir> [worlds] [fps]
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20
REMOTE=user@host SSH_OPTS='...' scripts/sync.sh '<command>'     # rsync to ~/batchplan on a GPU box and run there
```

Checking a LeRobot export with the real package requires a `.venv`, which is git-ignored:

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "lerobot[dataset,training]==0.6.1"
.venv/bin/python scripts/validate_lerobot.py data/lerobot_demo
.venv/bin/lerobot-train --dataset.repo_id=local/batchplan --dataset.root=data/lerobot_demo --policy.type=act \
  --policy.device=cpu --policy.push_to_hub=false --steps=50 --batch_size=16 --num_workers=0 --save_checkpoint=false --wandb.enable=false
```

## Design rules

These decisions are settled. Keep to them unless the user decides otherwise.

- **Batch-first.** Every query covers many items. Items reference worlds by index (`item_world`, `IkProblem.world`, `PlanProblem.world`), so one call spans many scenes.
- **One public handle, two hidden implementations.** `Device` (`device.rs`) is the only way to run batched work. `CpuBackend` and `GpuBackend` sit behind the crate-private `Backend` trait. Never make `Backend` public, and never add a way to call a backend that bypasses `Device`.
- **`Device` validates every batch** (array shapes, world indices) before either backend sees it, so malformed input is an `Err` on both devices. New `Device` entry points must go through `check_batch`.
- **Algorithms are separate modules over shared types.** `ik`, `trajopt`, `timing`, `datagen`, `npy` and `lerobot` are plain functions taking `&Device` and the `types` (`Pose`, `JointPaths`, `JointTrajectory`, `Solved`). There are no planner plugins and no runtime configuration.
  - Algorithm modules own seeding (the shared `rng::Rng`), validation and selection. Both devices therefore see bit-identical inputs.
- **Results carry their problems.** Use `IkResult::solved()` / `PlanResult::solved()`, which yield each problem with its best solution. Take the world from the problem, never from the problem's position in the list.
- **Small interfaces.** `Robot` exposes accessors and pose queries only; its internals are `pub(crate)`. Prefer deepening an existing module to adding a new public one.
- **Loaders stay thin.** A loader (`urdf.rs`) only translates a file into the crate-private `RobotDescription` (`description.rs`). Kinematics, sphere fitting and self-collision analysis work on the description, never on a file format.
- **Standalone, with optional bridges.**
  - The core has no middleware and reads no environment variables. Env vars appear only in tests and examples (`BATCHPLAN_REQUIRE_GPU`, `BENCH_LLVMPIPE`).
  - LeRobot export is behind the `lerobot` cargo feature, so the Arrow/Parquet dependencies stay optional.

## Architecture that spans files

**CPU/GPU twin.**
- `src/kernels.wgsl` and `src/cpu.rs` implement the same math function by function: `fk`, `collision`, `rot_log`, `chol6`, `ik_step`, the trajectory passes (`traj_samples`/`traj_sample_grad`, `traj_grad`) and the Adam update. `spline.rs` holds the B-spline basis that `basis` in WGSL mirrors.
- `cpu.rs` keeps index loops on purpose so the two read side by side.
- Any change to the math lands in both files in the same change. The parity tests in `tests/gpu.rs` and `tests/device.rs` catch drift.

**Host/shader structs.**
- Each struct the GPU reads is declared once with `shader_struct!` in `gpu.rs`: `GpuParams`, `GpuLink`, `GpuSphere`, `GpuObstacle` and `GpuIter`.
- Their WGSL declarations are generated from the same field list. `gpu.rs` builds the shader source as: generated prelude (`alias Vec4`, `MAX_DOF`, `MAX_LINKS`, `MAX_SPHERES`, `JAC_LEN`, the structs) + `kernels.wgsl`.
- To add a kernel parameter:
  1. Add a field to the right `shader_struct!`.
  2. Fill it in `GpuBackend`.
  3. Read it in WGSL as `P.<field>` (or `IT.<field>` for per-iteration values).
  4. Mirror it in `cpu.rs`.
- The kernel limits (`MAX_DOF` = 16, `MAX_JOINTS` = 16, `MAX_LINKS` = 32, `MAX_SPHERES` = 128) are defined in `robot.rs` and enforced when a robot loads.
- Per-joint kernel state (`jaxis`, `janchor`) is indexed by `Link.joint`, the moving-joint number, and sized `MAX_JOINTS`. Arrays that small stay in registers on AMD; indexing them per link (32 entries) cost 20% of GPU planning time.

**GPU details.**
- Bind group 0 has 11 storage buffers. `unmet_limits` skips adapters that can't provide them, and the error names each rejected adapter and what it lacks.
- Storage buffers are created by `storage::<T>()`, padded to at least one shader element. Empty obstacle lists previously crashed this way.
- Work is split per queue submission (`EVAL_CHUNK`, `IK_ITERS_PER_SUBMIT`, `TRAJ_ITERS_PER_SUBMIT`) to stay under driver watchdogs.
- Each trajopt iteration dispatches three passes:
  1. `traj_samples`: the collision gradient at every sample along every spline;
  2. `traj_grad`: per free control point, basis-weighted sample gradients plus smoothness;
  3. `traj_update`: Adam.

  `aux` holds [sample gradients | gradient | Adam m | Adam v]. Each iteration's Adam schedule comes from a dynamic-offset uniform (`GpuIter`).

**Planning flow.** In `trajopt::plan`:
1. Seed the paths: B-spline control points, the first three and last three pinned to start and goal. Seed 0's free points lie on the straight line; the others bend through random via points.
2. Run the backend's trajopt on the free control points.
3. Validate by sampling each spline densely (`validate_substeps` per span) and running `device.evaluate`.
4. `best()` picks the shortest valid seed.

`timing::Trajectory` times a path from bounds on its control-point differences (velocity, acceleration, jerk), so limits hold along the whole curve; `check` re-verifies them.

**Data pipeline.**
- `datagen::demonstrations` runs:
  1. IK with many seeds.
  2. Nominal plans from noisy, collision-free starts.
  3. `recovery_problems`: perturb solved paths and keep collision-free starts.
  4. Replanning from those starts.
  5. `Trajectory::new(..).sample(1 / dt)`: timing within the robot's limits at a random speed scale.
- `Origin::parent()` encodes recovery links for both exporters.
- `lerobot.rs` writes the v3.0 layout natively:
  - data, episode and task parquet; `tasks.parquet` carries pandas index metadata;
  - `info.json`, `stats.json`, and a `meta/batchplan.json` extension.
  - `action` is the next frame's state. `observation.environment_state` holds the goal plus padded obstacles.
  - Extension columns use names outside `observation.*` and `action*`, so LeRobot policies ignore them.

**Robot model.**
- `Robot::load` → `description::load_robot` (by extension) → `kinematics` (breadth-first tree, actuated joints numbered in that order, mimic joints resolved to their driving joint with a multiplier and offset, locked joints baked into fixed origins) → collision model (given, or fitted by `spheres.rs`) → SRDF pairs.
- Gradients and Jacobians walk `Link.chain`, a bitmask of the moving links at or above a link, root first, so mimic joints add into their leader. Both devices iterate it in the same order.
- `assets/franka/panda_collision.json` holds cuRobo's hand-tuned Panda spheres (Apache-2.0, credited in the README and `assets/README.md`); keep that credit when touching assets. `examples/common/mod.rs` has the Panda's options (locked fingers, `ee_link`, default pose).

## Working rules

- **A new test must fail on the unfixed code.** Prove it by planting the bug. For example, swapping `w_acc`/`w_vel` in the GPU packing must fail `trajopt_gradients_match_cpu_element_wise` and `trajopt_smoothness_gradient_matches_its_cost`.
- **Assert that test inputs actually exercise the code.** The smoothness test asserts its path isn't straight, because a straight path has zero smoothness gradient.
- **Test through the interface** (`Device`, `solve_ik`, `plan`, the exporters). Internal unit tests are only for internal seams, such as sphere pairs and `unmet_limits`.
- **Never compare CPU and GPU element-wise after Adam steps.** Adam normalizes each coordinate, which amplifies rounding noise in near-zero coordinates. Compare evaluations or gradients instead: `one_step_gradients` uses a huge `adam_epsilon` so that one step is plain gradient descent.
- **Performance claims need an A/B comparison on one machine.**
  - Build the previous commit in a `git worktree` with its own `CARGO_TARGET_DIR`, and alternate runs with the current code.
  - Check the load first; the dev machines often run other heavy work.
  - A single run proves nothing.
- **Keep `#[inline]` on hot functions the CPU backend calls across modules** (`Robot::fk`, `Fk::dpoint`, `box_distance`, `sphere_distance`). Without it, how the compiler splits the crate into chunks swings CPU throughput by 15–20%. Mark new hot cross-module helpers the same way.
- **`unwrap`/`expect` only for invariants you can prove.** Fallible paths return errors.
- **The repo is public.** Keep hostnames, usernames and account details out of committed files.
- **Comments only for what the next reader can't quickly recover from the code.** No comments referencing conversation context. If a workaround needs a paragraph of justification, fix the code instead.
