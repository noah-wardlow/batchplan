# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

batchplan is a Rust library for batched, collision-aware robot motion generation: IK and trajectory optimization over thousands of seeds, problems and worlds at once. It runs on any GPU wgpu supports (Vulkan, Metal, DX12) without CUDA or ROCm, and has a CPU implementation that produces the same numbers. One of its primary purposes is generating robot-learning training data: nominal demonstrations plus *recoveries* from perturbed states, exported as `.npy` arrays or LeRobot v3.0 datasets.

## Commands

```bash
cargo build --release --all-targets [--features lerobot] [--no-default-features]   # without `gpu`: CPU only, no wgpu
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                    # 94 tests; without the env var, GPU tests skip silently when no adapter exists
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features lerobot # + 5 export tests
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features usd     # + 7 OpenUSD tests
cargo test --release --test gpu trajopt_directions_match_cpu_element_wise   # one test (test files: cpu, gpu, device, export, pick_place, robot, trajectory, mjcf, usd, sdf, meshes, rrt, attach, threads)
cargo fmt --check                                               # rustfmt.toml: max_width 120
cargo clippy --release --all-targets [--features lerobot,usd | --no-default-features]   # zero warnings everywhere; CI denies them (lints in Cargo.toml)
cargo build --release --lib --target aarch64-unknown-linux-gnu --no-default-features     # the CPU-only robot build CI checks
cargo doc --no-deps --features lerobot,usd                      # keep at zero warnings
cargo run --release --example bench -- 512                      # GPU vs CPU throughput; BENCH_LLVMPIPE=1 adds the WGSL kernels on Mesa's CPU Vulkan driver
scripts/fetch_benchmark.sh && cargo run --release --example benchmark   # MotionBenchMaker + MπNets (2,600 Panda problems), GPU and CPU
cargo run --release --example datagen -- data/demo 512 20       # .npy dataset: <out_dir> [worlds] [fps]
cargo run --release --example depth [-- --cpu]                  # a distance grid from a rendered depth image, then plans around it
cargo run --release --example control_loop -- [seconds]         # a planner thread feeding a 50 Hz loop
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20   # --pick-place for pick-and-place, --file-mb N to split files sooner
REMOTE=user@host SSH_OPTS='...' scripts/sync.sh '<command>'     # rsync to ~/batchplan on a GPU box and run there
```

Checking USD loading against Pixar's `usd-core` (in the same `.venv`): `uv pip install --python .venv/bin/python usd-core && .venv/bin/python scripts/validate_usd.py`.

Checking a LeRobot export with the real package requires a `.venv`, which is git-ignored:

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "lerobot[dataset,training]==0.6.1"
.venv/bin/python scripts/validate_lerobot.py data/lerobot_demo
.venv/bin/lerobot-train --dataset.repo_id=local/batchplan --dataset.root=data/lerobot_demo --policy.type=act \
  --policy.device=cpu --policy.push_to_hub=false --steps=50 --batch_size=16 --num_workers=0 --save_checkpoint=false --wandb.enable=false --log_freq=10
```

## Design rules

These decisions are settled. Keep to them unless the user decides otherwise.

- **Batch-first.** Every query covers many items. Items reference worlds by index (`item_world`, `IkProblem.world`, `PlanProblem.world`), so one call spans many scenes.
- **Worlds live on the device.** `Device::upload(&[World]) -> Worlds` prepares worlds once (GPU buffers, CPU rotation matrices); every algorithm takes `&Worlds`. Each backend's `upload` returns its own form, which `Worlds::prepared` hands back; `Device` checks that worlds were uploaded to it. Exporters take `&[World]` (`worlds.as_slice()`). `Device::with_robot` shares the device id (and the GPU context or thread pool), so worlds stay valid when the robot changes, as with `Robot::attach`; attached objects are fixed links appended after every robot link.
- **One public handle, two hidden implementations.** `Device` (`device.rs`) is the only way to run batched work. `CpuBackend` and `GpuBackend` sit behind the crate-private `Backend` trait. Never make `Backend` public, and never add a way to call a backend that bypasses `Device`.
- **`Device` validates every batch** (array shapes, world indices, the device the worlds were uploaded to) before either backend sees it, so malformed input is an `Err` on both devices. New `Device` entry points must go through `check_batch`.
- **Algorithms are separate modules over shared types.** `ik`, `trajopt`, `timing`, `datagen`, `npy` and `lerobot` are plain functions taking `&Device`, `&Worlds` and the `types` (`Pose`, `JointPaths`, `JointTrajectory`, `Solved`). There are no planner plugins and no runtime configuration.
  - Algorithm modules own seeding (the shared `rng::Rng`), validation and selection. Both devices therefore see bit-identical inputs.
- **Results carry their problems.** Use `IkResult::solved()` / `PlanResult::solved()`, which yield each problem with its best solution. Take the world from the problem, never from the problem's position in the list.
- **Small interfaces.** `Robot` exposes accessors and pose queries only; its internals are `pub(crate)`. Prefer deepening an existing module to adding a new public one.
- **Loaders stay thin.** A loader (`urdf.rs`, `mjcf.rs`, `usd.rs` behind feature `usd`) only translates a file into the crate-private `Model` (`description.rs`): a `RobotDescription` and static scene shapes. Kinematics, sphere fitting, self-collision analysis and `World::load` work on those, never on a file format. Format semantics are translated, not approximated away: MJCF mesh geoms become `Geometry::ConvexHull` because MuJoCo collides their hulls.
- **Distance grids err toward collision.** `sdf.rs` builders store at most the true signed distance per point, minus half a voxel diagonal (the most trilinear interpolation overestimates by). Keep that invariant when touching a builder; `tests/sdf.rs` checks it.
- **Typed errors at the surface.** Public functions return `batchplan::Error` (`error.rs`); its kinds are what a host matches on, so choose the kind deliberately (`ensure_input!`/`input!` for bad input). The format loaders and robot building keep `anyhow` internally for context chains; `Robot::load`, `World::load` and `CollisionModel::load` turn them into `Error::Load` with the path.
- **Embeddable.** The CPU device owns its rayon pool (`Device::cpu_threads`); every parallel loop in `cpu.rs` runs inside `self.pool.install`, never on the global pool (`tests/threads.rs` counts OS threads). `PlanOptions::time_budget` becomes a deadline passed to `Backend::trajopt` and to the fallback's `connect_until`/`shortcut_until`; without one, nothing waits or checks the clock, and results are deterministic.
- **USD stays optional.** The `openusd` crates are pinned exactly (`=0.7.0`, pre-1.0) and only built with feature `usd`.
- **Standalone, with optional bridges.**
  - The core has no middleware and reads no environment variables. Env vars appear only in tests and examples (`BATCHPLAN_REQUIRE_GPU`, `BENCH_LLVMPIPE`).
  - LeRobot export is behind the `lerobot` cargo feature, so the Arrow/Parquet dependencies stay optional.

## Architecture that spans files

**CPU/GPU twin.**
- The GPU kernels (`src/kernels.wgsl` plus the per-robot code `robot_wgsl` writes in `gpu.rs`) and `src/cpu.rs` implement the same math function by function: `fk`, `collision`, `rot_log`, `chol6`, `ik_step`, the trajectory passes (`traj_costs`/`traj_cost`, `traj_search`, `traj_samples`/`traj_sample_grad`, `traj_grad`, `lbfgs_direction`). `spline.rs` holds the B-spline basis that `basis` in WGSL mirrors; `world.rs` and `sdf.rs` (`grid_distance`) hold the obstacle distances that `obstacle_distance` and `grid_distance` mirror.
- `cpu.rs` keeps index loops on purpose so the two read side by side.
- Any change to the math lands in both files in the same change. The parity tests in `tests/gpu.rs` and `tests/device.rs` catch drift.
- Grid building has its own twin: `src/grids.wgsl` mirrors `sdf.rs`'s `DepthImage::classify`, `depth_occupancy`, `integrate`, `squared_edt`/`Line`/`meet` and `finish`. Builders (`SdfGrid::from_points`/`from_depth`, `OccupancyMap::integrate`/`grid`) take a `Device` and go through `Backend::grid_values` and `Backend::integrate`.
  - The distance transform is integer arithmetic, so both devices build identical grids from the same occupancy (`every_device_builds_the_same_grids_from_occupancy`).
  - Each value is finished by a table the host builds (`finishing_table`, sized by the grid's squared diagonal, hence `MAX_DIAGONAL`), so the GPU never needs `sqrt` to match the CPU.
  - Classification projects voxel centres in floating point and may differ at rounding ties. The internal test `every_device_classifies_depth_images_alike` bounds that at 0.05% of voxels; the behavioural tests in `tests/sdf.rs` run on every device.
  - The GPU keeps its grid buffers (`GridBuffers`) between builds, because wgpu zero-fills new buffers.

**Host/shader structs.**
- Each struct the GPU reads is declared once with `shader_struct!` in `gpu.rs`: `GpuParams`, `GpuLink`, `GpuSphere` and `GpuObstacle`. So are the constants (`MAX_HISTORY`, the line-search steps `LINE_SEARCH`, obstacle kinds).
- Their WGSL declarations are generated from the same field list. `gpu.rs` builds the shader source as: generated prelude (`alias Vec4`, the constants, the structs) + `robot_wgsl(robot)` + `kernels.wgsl`.
- To add a kernel parameter:
  1. Add a field to the right `shader_struct!`.
  2. Fill it in `GpuBackend`.
  3. Read it in WGSL as `P.<field>`.
  4. Mirror it in `cpu.rs`.
- There are no fixed size limits on robots.
  - The CPU kernels keep per-configuration state in `Scratch` (`cpu.rs`: FK frames, sphere centres, wrenches, IK buffers). It is sized to the robot and reused per thread (`for_each_init`, or one per path), so hot loops neither allocate nor clear more than the robot needs. Inline small vectors were tried first and cost 5–27% of CPU throughput.
  - The GPU kernels are written per robot. Self-collision is unrolled per link pair up to `UNROLLED_LINK_PAIRS` (64); beyond that, a run-time loop over frames copied into arrays keeps the generated code growing with links, not pairs.
- Links no joint moves (`Link.chain` empty: the base and what is fixed to it) are not checked against the world on either device, nor by `MeshModel`: their contact with the scene does not depend on the configuration, and a mounted robot's base rests on its table. They still self-collide.
- World collision is gated per link: an obstacle farther from a link's bounding sphere (`link_bounds`, `GpuLink.bound`) than `max(margin, 0)` skips the link's spheres (`Robot::sphere_ranges`), and the gap stands in for their clearance. Distance grids are never gated: their interpolated distance is not 1-Lipschitz.
- Self-collision sphere pairs are grouped by link pair (`Robot::self_link_pairs`). A link pair whose bounding spheres (`link_bounds`, `GpuLink.bound`) are farther apart than `max(self_margin, 0)` skips its sphere pairs, and its gap stands in for its clearance. On the GPU, the `pairs` buffer starts with one `(first, count)` entry per link pair, followed by the sphere pairs.

**Per-robot GPU kernels.**
- `robot_wgsl` writes the robot-specific code: `MAX_DOF`, forward kinematics, the end effector's Jacobian and the collision cost.
  - Each link's frame (`rot_i`, `pos_i`) and collision wrench (`force_i`, `moment_i`) is its own private variable, indexed only by constants.
  - Sphere and pair ranges come from the buffers (`GpuLink.first_sphere`, the `pairs` header), so robots that differ only in their spheres share code.
- Why: Mesa's RADV moves any private array indexed at run time and larger than 256 bytes into scratch memory. With per-link and per-sphere arrays, every collision kernel spilled 3–5 KB per invocation. Writing the code per robot removed that and made GPU planning 4.8× faster on the Radeon and 2.7× on the M4 Pro. Never add a run-time-indexed private array larger than that to a collision kernel. To check on RADV, `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats cargo run --release --example bench -- 8` prints each pipeline's VGPRs and scratch size.
- Collision gradients accumulate a wrench per link (force, and moment about the world origin). One reverse pass, leaves first, adds each link's wrench into its parent and gives each joint `a·(M − o×F)` (revolute) or `a·F` (prismatic). `cpu.rs` does the same.
- `GpuBackend` compiles kernels per robot shape and caches them by generated source, shared by every backend `with_robot` derives.
- **Lanes.** The collision kernels (`evaluate_main`, `clearance_main`, `ik_main`, `traj_costs`, `traj_samples`) give each configuration `LANES` consecutive invocations. `LANES` is a WGSL `override`, and each kernel is built twice:
  - with 1 lane for large batches, where the lane code folds away;
  - with `SHARED_LANES` (8) for batches whose configurations times 8 fit within `LANE_TARGET` threads.

  Lanes split the inner sphere loops of `collision()`. Every lane runs every block, so a wave stays converged; splitting whole blocks across lanes diverged and was slower. `group_collision` and `group_grad` combine the lanes. These kernels have no early returns, because lanes meet at barriers. `evaluation_does_not_depend_on_the_batch` checks that both builds agree. Pipelines skip workgroup zero-initialization: every kernel writes its workgroup memory before reading it.

**GPU details.**
- Bind group 0 has 12 storage buffers. `unmet_limits` skips adapters that can't provide them, and the error names each rejected adapter and what it lacks.
- `GpuWorlds` holds every world's obstacles, per-world ranges and `grid_data`: each distinct grid (by `Arc` pointer) once, two half floats per `u32`, read with `unpack2x16float`. Grids store subnormal halves as zero because GPUs may flush them.
- Storage buffers are created by `storage::<T>()`, padded to at least one shader element. Empty obstacle lists previously crashed this way.
- Work is split per queue submission (`EVAL_CHUNK`, `IK_ITERS_PER_SUBMIT`, `TRAJ_ITERS_PER_SUBMIT`) to stay under driver watchdogs.
- Trajectory optimization is L-BFGS with a parallel line search. Each round dispatches five passes:
  1. `traj_costs`: the collision cost at every sample of every path moved by each line-search step;
  2. `traj_search` (one workgroup per path): the cheapest step moves the path if it lowers the cost, otherwise the history resets;
  3. `traj_samples`: the collision gradient at every sample;
  4. `traj_grad`: per free control point, basis-weighted sample gradients plus smoothness;
  5. `lbfgs_direction` (one workgroup per path): record the last step and gradient change, then the two-loop recursion.

  The first round only prices the seeds (their direction is still zero). The per-path passes run one workgroup per path: each invocation owns every `WORKGROUP`-th element (coalesced loads), and sums go through `workgroup_sum`/`workgroup_max`, whose results come back through `workgroupUniformLoad` so later barriers stay in uniform control flow. `aux` holds each path's L-BFGS state (`lbfgs_stride`, mirrored by the `Lbfgs` struct in `cpu.rs`); paths are optimized in chunks whose state fits one storage binding.

**Planning flow.** In `trajopt::plan`:
1. Seed the paths: B-spline control points, the first three and last three pinned to start and goal. Goals come from `Robot::goal_variants`: continuous joints turned the short way, and joints wider than a turn at their nearest equivalent, then a turn further either way. Seed 0 runs straight to the nearest variant, odd seeds straight to the others while they last, and the rest bend toward the nearest through random via points.
2. Run the backend's trajopt on the free control points.
3. Validate by sampling each spline densely (`validate_substeps` per span) and running `Device::clearance` (crate-private: world and self clearance without cost or gradient; IK, RRT and datagen use it too).
4. `best()` picks the shortest valid seed.
5. Problems without a valid seed fall back (`PlanOptions::fallback`): `rrt::connect` (RRT-Connect toward every goal variant, several extensions per problem per round, every problem's edges checked in one `Device::clearance`; continuous joints step and measure the short way, `Robot::turn_toward`, and the found path is unwrapped), `shortcut::shortcut`, then `trace` turns the waypoints into control points whose spline runs exactly along them (each waypoint tripled). The traced spline and its optimized version are validated; the shorter valid one replaces seed 0.

`timing::Trajectory` is the path plus a time map `σ(t)` (a uniform cubic B-spline in time). `Trajectory::new` keeps the faster of uniform timing (`σ = t / h`, `h` from control-point differences, exact along the whole curve) and `topp::time_map`. The latter is TOPP-RA on the path parameter, smoothed (Schoenberg), with caps lowered locally where the smoothed map overshoots and a final stretch measured at `CHECK_SAMPLES` per map span. `check` is exact for uniform maps and samples `CHECK_SAMPLES` per span otherwise. Moving starts (`PlanProblem::start_motion`):
- `trajopt::continue_motion` pins the first three control points for a knot interval `h0`;
- `Trajectory::moving` recovers `h0` from them and times the path from ṡ = 1/h0. Stage 0 is held to the full limits, candidates are refined at full density against limits less the margin, and nothing is stretched;
- `check_from` checks the start state.

**Data pipeline.**
- `datagen::demonstrations` runs:
  1. IK with many seeds.
  2. Nominal plans from noisy, collision-free starts.
  3. `recovery_problems`: perturb solved paths and keep collision-free starts.
  4. Replanning from those starts.
  5. `Trajectory::new(..).sample(1 / dt)`: timing within the robot's limits at a random speed scale.
- `datagen::pick_and_place` chains segments per task: approach (the object an obstacle), descend (worlds without the object, at index `tasks + i`), close, lift (the descent reversed), transfer (on a `with_robot` device holding the object, one per object size), lower, open, retreat (the lowering reversed). IK for the grasp and the place pose is seeded from the configuration above them and takes `IkResult::nearest`. `Episode` concatenates rest-to-rest segments, dropping each one's duplicate first sample.
- `Demonstration` carries `task`, the gripper's opening per sample and an optional `Carried` obstacle with its pose per sample; `check_demos` checks their lengths.
- `Origin::parent()` encodes recovery links for both exporters.
- `lerobot.rs` writes the v3.0 layout natively:
  - data and episode metadata through `Files`, which rolls over past `data_files_size_in_mb`; each episode row records its own file. Global stats are `Aggregate`d from per-episode stats as LeRobot's `aggregate_feature_stats` does, so nothing holds every frame. Keep features in column order (`serde_json::Map` sorts its keys);
  - `tasks.parquet` lists the distinct tasks with pandas index metadata; `info.json`, `stats.json`, and a `meta/batchplan.json` extension.
  - The state is the joints plus the gripper; `action` is the next frame's state. `observation.environment_state` holds the goal plus padded obstacles, each where it is at that frame.
  - Extension columns use names outside `observation.*` and `action*`, so LeRobot policies ignore them.

**Robot model.**
- `Robot::load` → `description::load_robot` (by extension) → if the description has loops, `loops::close_loops` on the open tree (each loop's passive joints become quartic mimics of its one actuated joint: the least spring energy that closes the loop within their limits, by projected Levenberg-Marquardt over the driver's range; `tests/mjcf.rs` holds MuJoCo's settled 2F-85 as the reference) → `kinematics` (planar, ball and floating joints expanded into one-axis joints on massless links by `expand_compound_joints`, so nothing downstream sees them; breadth-first tree, actuated joints numbered in that order, mimic joints resolved to their driving joint with a `Curve` (a polynomial of degree ≤ 4, composed along mimic chains; `Fk` records each nonlinear joint's slope, which gradients and Jacobians use in place of the multiplier), locked joints baked into fixed origins) → collision model (given, or fitted by `spheres.rs`) → SRDF pairs.
- Continuous joints (`Robot::continuous`) are unbounded reals whose values wrap every 2π; `lower`/`upper` give them only the turn that seeds and samples come from. Clamp and check with `Robot::bounds`/`Robot::within`, never with `lower`/`upper` directly; the GPU limits buffer holds ±∞ for them.
- IK Jacobians walk `Link.chain`, a bitmask of the moving links at or above a link, root first, so mimic joints add into their leader. Both devices iterate it in the same order (`robot_wgsl` unrolls it).
- Each `Link` keeps its collision shapes (`Link.shapes`, from the description or an attached object). `MeshModel` (`meshes.rs`) loads them as parry3d meshes and convex hulls and measures exact mesh clearances by branch and bound; it is a CPU post-check, not part of either device.
- `assets/franka/panda_collision.json` holds cuRobo's hand-tuned Panda spheres (Apache-2.0, credited in the README and `assets/README.md`); keep that credit when touching assets. `examples/common/mod.rs` has the Panda's options (locked fingers, `ee_link`, default pose).

## Working rules

- **A new test must fail on the unfixed code.** Prove it by planting the bug. For example, swapping `w_acc`/`w_vel` in the GPU packing must fail `trajopt_directions_match_cpu_element_wise` and `trajopt_smoothness_gradient_matches_its_cost`.
- **Assert that test inputs actually exercise the code.** The smoothness test asserts its path isn't straight, because a straight path has zero smoothness gradient.
- **Test through the interface** (`Device`, `solve_ik`, `plan`, the exporters). Internal unit tests are only for internal seams, such as sphere pairs and `unmet_limits`.
- **Compare optimizer steps by direction, not position.** The line search scales each L-BFGS step, so `step_directions` in `tests/gpu.rs` scales each path's step to its largest entry: step 1 is the negative gradient, later steps the two-loop recursion. Outcomes are compared through the full trajectory cost (`trajopt_lowers_the_cost_as_far_as_the_cpu`).
- **Performance claims need an A/B comparison on one machine.**
  - Build the previous commit in a `git worktree` with its own `CARGO_TARGET_DIR`, and alternate runs with the current code.
  - Check the load first; the dev machines often run other heavy work.
  - A single run proves nothing.
- **Keep `#[inline]` on hot functions the CPU backend calls across modules** (`Robot::fk_into`, `Fk::dpoint`, `box_distance`, `sphere_distance`). Without it, how the compiler splits the crate into chunks swings CPU throughput by 15–20%. Mark new hot cross-module helpers the same way. `cpu::collision` is `#[inline(always)]`: with a second cost-only caller, LLVM stopped inlining it into the line search (−5%).
- **`unwrap`/`expect` only for invariants you can prove.** Fallible paths return errors.
- **The repo is public.** Keep hostnames, usernames and account details out of committed files.
- **Comments only for what the next reader can't quickly recover from the code.** No comments referencing conversation context. If a workaround needs a paragraph of justification, fix the code instead.
