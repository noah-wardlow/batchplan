//! Planner output as policy-training demonstrations: nominal reaches plus recoveries from
//! perturbed states, timed within the robot's limits at randomized speeds.

use glam::{Quat, Vec3};

use crate::device::{Device, Worlds, collision_free};
use crate::error::{Result, ensure_input, input};
use crate::ik::{IkOptions, IkProblem, IkResult, solve_ik};
use crate::rng::Rng;
use crate::robot::AttachedObject;
use crate::robot::Robot;
use crate::spline;
use crate::timing::Trajectory;
use crate::trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
use crate::types::{JointTrajectory, Pose, Solved};
use crate::world::{Obstacle, World};

#[derive(Clone, Copy, Debug)]
pub struct RecoveryOptions {
    pub per_trajectory: usize,
    /// Standard deviation of the joint-space perturbation (radians).
    pub sigma: f32,
    /// Range of path fractions where perturbations happen.
    pub phase: (f32, f32),
    pub rng_seed: u64,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self { per_trajectory: 2, sigma: 0.15, phase: (0.2, 0.8), rng_seed: 3 }
    }
}

/// A planning problem that starts from a perturbed state of a planned path.
#[derive(Clone, Debug)]
pub struct Recovery {
    /// Index in `result.problems` of the problem whose best path was perturbed.
    pub parent: usize,
    /// Fraction of the nominal path where the perturbation happened.
    pub phase: f32,
    pub problem: PlanProblem,
}

/// Simulates policy mistakes: perturbs states along the best path of each solved problem and
/// keeps the collision-free ones as problems toward the same goal. Planning them yields
/// demonstrations of recovering from off-nominal states, which raw planner output never contains.
pub fn recovery_problems(
    device: &Device,
    worlds: &Worlds,
    result: &PlanResult,
    o: &RecoveryOptions,
) -> Result<Vec<Recovery>> {
    let robot = device.robot();
    let n = robot.dof();
    let mut rng = Rng::new(o.rng_seed);
    let mut candidates = vec![];
    let mut on_path = vec![0.0; n];
    for Solved { index: parent, problem, solution: path } in result.solved() {
        for _ in 0..o.per_trajectory {
            let phase = rng.range(o.phase.0, o.phase.1);
            spline::at_phase(path, n, phase, &mut on_path);
            let start: Vec<f32> = (0..n)
                .map(|j| (on_path[j] + o.sigma * rng.normal()).clamp(robot.bounds(j).0, robot.bounds(j).1))
                .collect();
            let next = PlanProblem { world: problem.world, start, goal: problem.goal.clone(), start_motion: None };
            candidates.push(Recovery { parent, phase, problem: next });
        }
    }
    let starts: Vec<f32> = candidates.iter().flat_map(|c| c.problem.start.iter().copied()).collect();
    let item_world: Vec<u32> = candidates.iter().map(|c| c.problem.world).collect();
    let clear = device.clearance(worlds, &item_world, &starts)?;
    Ok(candidates.into_iter().enumerate().filter(|&(i, _)| collision_free(clear[i])).map(|(_, c)| c).collect())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Origin {
    Nominal,
    /// Replanned from a perturbed state of demonstration `parent` (an index into the same output),
    /// `phase` of the way along it.
    Recovery {
        parent: usize,
        phase: f32,
    },
}

impl Origin {
    /// The demonstration a recovery branches from; `None` for nominal demonstrations.
    pub fn parent(&self) -> Option<usize> {
        match *self {
            Origin::Nominal => None,
            Origin::Recovery { parent, .. } => Some(parent),
        }
    }
}

/// The task of every reach demonstration.
pub const REACH_TASK: &str = "Move the gripper to the target pose.";

/// One time-parameterized episode in `worlds[world]`.
#[derive(Clone, Debug)]
pub struct Demonstration {
    pub origin: Origin,
    pub world: u32,
    /// What the episode does, in words; exporters index the distinct tasks.
    pub task: String,
    /// The pose the episode drives toward: the IK frame's for a reach, the object's for
    /// pick-and-place.
    pub goal: Pose,
    pub trajectory: JointTrajectory,
    /// The gripper's opening at each sample: 1 open, 0 closed.
    pub gripper: Vec<f32>,
    /// An obstacle of the world that moves during the episode, such as a carried object.
    pub carried: Option<Carried>,
}

/// An obstacle that moves during an episode: its index among the world's obstacles and its pose
/// (the obstacle's own frame: centre and rotation) at each sample.
#[derive(Clone, Debug)]
pub struct Carried {
    pub obstacle: usize,
    pub poses: Vec<Pose>,
}

#[derive(Clone, Copy, Debug)]
pub struct DemoOptions {
    pub ik: IkOptions,
    pub plan: PlanOptions,
    pub recovery: RecoveryOptions,
    /// Standard deviation of the joint noise added to the default pose for start states (radians).
    /// Starts that collide fall back to the default pose.
    pub start_noise: f32,
    /// Each demonstration's speed scale (a fraction of the fastest timing within the robot's
    /// limits) is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    pub rng_seed: u64,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            recovery: RecoveryOptions::default(),
            start_noise: 0.3,
            speed_scale: (0.6, 1.0),
            dt: 0.05,
            rng_seed: 7,
        }
    }
}

/// Reach-to-pose demonstrations for each goal, followed by recoveries from perturbed states of
/// those demonstrations. Goals without a collision-free IK solution or plan are skipped.
pub fn demonstrations(
    device: &Device,
    worlds: &Worlds,
    goals: &[IkProblem],
    o: &DemoOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let n = robot.dof();
    let (slowest, fastest) = o.speed_scale;
    ensure_input!(
        slowest > 0.0 && slowest <= fastest && fastest <= 1.0,
        "speed scales must satisfy 0 < slowest <= fastest <= 1, got {:?}",
        o.speed_scale
    );
    ensure_input!(o.dt.is_finite() && o.dt > 0.0, "dt must be positive, got {}", o.dt);
    let mut rng = Rng::new(o.rng_seed);
    let ik = solve_ik(device, worlds, goals, &o.ik)?;

    let mut starts: Vec<f32> = (0..goals.len())
        .flat_map(|_| {
            (0..n)
                .map(|j| {
                    (robot.default_q[j] + o.start_noise * rng.normal()).clamp(robot.bounds(j).0, robot.bounds(j).1)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = goals.iter().map(|g| g.world).collect();
    let start_clear = device.clearance(worlds, &item_world, &starts)?;
    for g in (0..goals.len()).filter(|&g| !collision_free(start_clear[g])) {
        starts[g * n..(g + 1) * n].copy_from_slice(&robot.default_q);
    }
    // Nominal plan problem p reaches the target of IK problem goal_of[p].
    let (goal_of, problems): (Vec<usize>, Vec<PlanProblem>) = ik
        .solved()
        .map(|s| {
            let start = starts[s.index * n..(s.index + 1) * n].to_vec();
            (s.index, PlanProblem { world: s.problem.world, start, goal: s.solution.to_vec(), start_motion: None })
        })
        .unzip();
    let nominal = plan(device, worlds, &problems, &o.plan)?;
    let recoveries = recovery_problems(device, worlds, &nominal, &o.recovery)?;
    let recovery_plans: Vec<PlanProblem> = recoveries.iter().map(|r| r.problem.clone()).collect();
    let recovered = plan(device, worlds, &recovery_plans, &o.plan)?;

    let mut timed = |path: &[f32]| {
        let speed_scale = rng.range(o.speed_scale.0, o.speed_scale.1);
        Trajectory::new(robot, path, speed_scale)?.sample(1.0 / o.dt)
    };
    let reach = |origin, world, goal, trajectory: JointTrajectory| Demonstration {
        origin,
        world,
        task: REACH_TASK.into(),
        goal,
        gripper: vec![1.0; trajectory.len()],
        trajectory,
        carried: None,
    };
    let mut demos = vec![];
    // Demonstration index of each solved nominal problem, for recovery parents.
    let mut demo_of = vec![usize::MAX; problems.len()];
    for s in nominal.solved() {
        demo_of[s.index] = demos.len();
        let goal = goals[goal_of[s.index]].target;
        demos.push(reach(Origin::Nominal, s.problem.world, goal, timed(s.solution)?));
    }
    for s in recovered.solved() {
        let rec = &recoveries[s.index];
        let parent = demo_of[rec.parent];
        let goal = demos[parent].goal;
        demos.push(reach(Origin::Recovery { parent, phase: rec.phase }, s.problem.world, goal, timed(s.solution)?));
    }
    Ok(demos)
}

/// The shared sample period of demonstrations to export, after checking that each one belongs to
/// one of `worlds`, has the robot's joints, holds whole samples with a gripper opening each and
/// any carried obstacle's pose each, and shares that period.
pub(crate) fn check_demos(robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<f32> {
    ensure_input!(!demos.is_empty(), "no demonstrations to export");
    let dt = demos[0].trajectory.dt;
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        ensure_input!((d.world as usize) < worlds.len(), "demonstration {i} is in world {}, past the end", d.world);
        ensure_input!(t.dof == robot.dof(), "demonstration {i} has {} joints, the robot has {}", t.dof, robot.dof());
        ensure_input!(
            !t.positions.is_empty() && t.positions.len() == t.velocities.len() && t.positions.len() % t.dof == 0,
            "demonstration {i} does not hold whole samples"
        );
        ensure_input!(t.dt == dt, "all demonstrations must share one dt");
        ensure_input!(
            d.gripper.len() == t.len() && d.gripper.iter().all(|g| (0.0..=1.0).contains(g)),
            "demonstration {i} needs one gripper opening in [0, 1] per sample"
        );
        if let Some(c) = &d.carried {
            let obstacles = worlds[d.world as usize].obstacles.len();
            ensure_input!(c.obstacle < obstacles, "demonstration {i} carries obstacle {} of {obstacles}", c.obstacle);
            ensure_input!(c.poses.len() == t.len(), "demonstration {i} needs one carried pose per sample");
        }
    }
    Ok(dt)
}

/// One pick-and-place task: pick up the cuboid `object` of `worlds[world]` from above and set it
/// down with its centre and rotation at `place`. Both must be upright (their z axis vertical).
#[derive(Clone, Debug)]
pub struct PickPlace {
    pub world: u32,
    pub object: usize,
    pub place: Pose,
}

#[derive(Clone, Copy, Debug)]
pub struct PickPlaceOptions {
    pub ik: IkOptions,
    pub plan: PlanOptions,
    /// How far above the grasp and the place pose the gripper approaches from and retreats to
    /// (meters).
    pub approach: f32,
    /// How far below the object's top the gripper holds it (meters).
    pub grasp_depth: f32,
    /// The widest object the gripper closes on (meters).
    pub max_width: f32,
    /// How long closing or opening the gripper takes (seconds).
    pub gripper_time: f32,
    /// Each segment's speed scale is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    pub rng_seed: u64,
}

impl Default for PickPlaceOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            approach: 0.1,
            grasp_depth: 0.02,
            max_width: 0.08,
            gripper_time: 0.5,
            speed_scale: (0.6, 1.0),
            dt: 0.05,
            rng_seed: 11,
        }
    }
}

/// Pick-and-place demonstrations, one per task that every step succeeds for; the others are
/// skipped. The robot's IK frame is the grasp point, its z axis pointing out of the gripper and its
/// y axis along the fingers' closing direction (as the Panda's `ee_link` in the examples).
///
/// Each episode chains planned segments, all from rest to rest: approach a pose above the object
/// (which is an obstacle) from the default configuration; descend to the grasp (the object no
/// longer an obstacle, as the fingers close around it); close the gripper; lift back up; transfer
/// to above the place pose holding the object (`Robot::attach`, touching only the links fixed to
/// the wrist); lower it; open; retreat. The grasp is top-down, across the object's narrower side.
/// The demonstration records the gripper's opening and where the object is at every sample, and
/// a task such as "Pick up the box and put it 20 cm to the left."
pub fn pick_and_place(
    device: &Device,
    worlds: &[World],
    tasks: &[PickPlace],
    o: &PickPlaceOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let n = robot.dof();
    let (slowest, fastest) = o.speed_scale;
    ensure_input!(
        slowest > 0.0 && slowest <= fastest && fastest <= 1.0,
        "speed scales must satisfy 0 < slowest <= fastest <= 1, got {:?}",
        o.speed_scale
    );
    let positive = |v: f32| v.is_finite() && v > 0.0;
    ensure_input!(
        positive(o.dt)
            && positive(o.approach)
            && positive(o.max_width)
            && o.gripper_time >= 0.0
            && o.grasp_depth >= 0.0,
        "pick-and-place needs positive dt, approach and max_width and non-negative gripper_time and grasp_depth"
    );
    let upright = |r: Quat| (r * Vec3::Z).dot(Vec3::Z) > 1.0 - 1e-4;
    let mut grasps = vec![];
    for (i, t) in tasks.iter().enumerate() {
        let world =
            worlds.get(t.world as usize).ok_or_else(|| input!("task {i}: world {} of {}", t.world, worlds.len()))?;
        let Some(&Obstacle::Cuboid { center, half_extents: half, rotation }) = world.obstacles.get(t.object) else {
            return Err(input!("task {i}: obstacle {} of world {} is not a cuboid", t.object, t.world));
        };
        ensure_input!(
            upright(rotation) && upright(t.place.rotation),
            "task {i}: the object and the place pose must be upright"
        );
        ensure_input!(2.0 * half.x.min(half.y) <= o.max_width, "task {i}: the object is wider than the gripper opens");
        grasps.push(Grasp::new(Pose { position: center, rotation }, half, o.grasp_depth));
    }

    // Each task's world as given (index i) and without its object (index tasks + i).
    let without = |t: &PickPlace| {
        let mut w = worlds[t.world as usize].clone();
        w.obstacles.remove(t.object);
        w
    };
    let scene: Vec<World> =
        tasks.iter().map(|t| worlds[t.world as usize].clone()).chain(tasks.iter().map(without)).collect();
    let uploaded = device.upload(&scene)?;
    let clear = |i: usize| (tasks.len() + i) as u32;
    let up = Vec3::Z * o.approach;
    let at = |p: Pose, lift: Vec3| Pose { position: p.position + lift, rotation: p.rotation };
    let place_tool: Vec<Pose> = tasks.iter().zip(&grasps).map(|(t, g)| g.tool_for(t.place)).collect();

    // Above the object, then the grasp from there.
    let above: Vec<IkProblem> = grasps
        .iter()
        .enumerate()
        .map(|(i, g)| IkProblem { world: i as u32, target: at(g.tool, up), seed: None })
        .collect();
    let above = solve_ik(device, &uploaded, &above, &o.ik)?;
    let mut ok: Vec<Option<[Vec<f32>; 4]>> = (0..tasks.len()).map(|_| None).collect();
    let seeded = |ik: &IkResult, i: usize, from: &[f32]| ik.nearest(i, from).map(<[f32]>::to_vec);
    let grasp: Vec<IkProblem> = grasps
        .iter()
        .enumerate()
        .map(|(i, g)| IkProblem { world: clear(i), target: g.tool, seed: above.best(i).map(<[f32]>::to_vec) })
        .collect();
    let grasp = solve_ik(device, &uploaded, &grasp, &o.ik)?;

    // Holding each distinct object size (the grasp fixes where the object sits in the IK frame).
    let ee = robot.ee_link;
    let touch: Vec<String> =
        robot.links.iter().filter(|l| l.chain == robot.links[ee].chain).map(|l| l.name.clone()).collect();
    let mut holders: Vec<(Vec3, Device)> = vec![];
    for g in &grasps {
        if holders.iter().all(|(half, _)| *half != g.half) {
            let held = robot.attach(&AttachedObject {
                name: "held object".into(),
                link: robot.links[ee].name.clone(),
                shapes: vec![Obstacle::Cuboid {
                    center: g.object_in_tool.position,
                    half_extents: g.half,
                    rotation: g.object_in_tool.rotation,
                }],
                touch_links: touch.clone(),
                spheres: Default::default(),
            })?;
            holders.push((g.half, device.with_robot(&held)?));
        }
    }

    // Above the place pose (holding the object), then the place pose from there.
    let mut above_place = vec![None; tasks.len()];
    for (half, held) in &holders {
        let group: Vec<usize> = (0..tasks.len()).filter(|&i| grasps[i].half == *half).collect();
        let problems: Vec<IkProblem> = group
            .iter()
            .map(|&i| IkProblem {
                world: clear(i),
                target: at(place_tool[i], up),
                seed: above.best(i).map(<[f32]>::to_vec),
            })
            .collect();
        let ik = solve_ik(held, &uploaded, &problems, &o.ik)?;
        for (k, &i) in group.iter().enumerate() {
            above_place[i] = above.best(i).and_then(|from| seeded(&ik, k, from));
        }
    }
    let place: Vec<IkProblem> = (0..tasks.len())
        .map(|i| IkProblem { world: clear(i), target: place_tool[i], seed: above_place[i].clone() })
        .collect();
    let place = solve_ik(device, &uploaded, &place, &o.ik)?;
    for i in 0..tasks.len() {
        let q1 = above.best(i).map(<[f32]>::to_vec);
        let q2 = q1.as_deref().and_then(|q1| seeded(&grasp, i, q1));
        let q3 = above_place[i].clone();
        let q4 = q3.as_deref().and_then(|q3| seeded(&place, i, q3));
        if let (Some(q1), Some(q2), Some(q3), Some(q4)) = (q1, q2, q3, q4) {
            ok[i] = Some([q1, q2, q3, q4]);
        }
    }

    // The planned segments, in batches over the tasks still going: approach, descend, transfer
    // (holding the object, one batch per object size) and lower.
    let start = robot.default_q.clone();
    let mut paths: [Vec<Vec<f32>>; 4] = Default::default();
    let mut live: Vec<usize> = (0..tasks.len()).filter(|&i| ok[i].is_some()).collect();
    for (s, segment) in paths.iter_mut().enumerate() {
        *segment = vec![vec![]; tasks.len()];
        let ends = |i: usize| {
            let q = ok[i].as_ref().expect("live tasks have their configurations");
            match s {
                0 => (start.clone(), q[0].clone()),
                1 => (q[0].clone(), q[1].clone()),
                2 => (q[0].clone(), q[2].clone()),
                _ => (q[2].clone(), q[3].clone()),
            }
        };
        let groups: Vec<(&Device, Vec<usize>)> = if s == 2 {
            holders
                .iter()
                .map(|(half, held)| (held, live.iter().copied().filter(|&i| grasps[i].half == *half).collect()))
                .collect()
        } else {
            vec![(device, live.clone())]
        };
        let mut next = vec![];
        for (d, group) in groups {
            let problems: Vec<PlanProblem> = group
                .iter()
                .map(|&i| {
                    let (start, goal) = ends(i);
                    PlanProblem { world: if s == 0 { i as u32 } else { clear(i) }, start, goal, start_motion: None }
                })
                .collect();
            for solved in plan(d, &uploaded, &problems, &o.plan)?.solved() {
                segment[group[solved.index]] = solved.solution.to_vec();
                next.push(group[solved.index]);
            }
        }
        next.sort_unstable();
        live = next;
    }

    let mut rng = Rng::new(o.rng_seed);
    let mut timed = |path: &[f32]| {
        let speed_scale = rng.range(o.speed_scale.0, o.speed_scale.1);
        Trajectory::new(robot, path, speed_scale)?.sample(1.0 / o.dt)
    };
    let hold = ((o.gripper_time / o.dt).round() as usize).max(1);
    let mut demos = vec![];
    for i in live {
        let [approach, descend, transfer, lower] = [&paths[0][i], &paths[1][i], &paths[2][i], &paths[3][i]];
        let (approach, descend, transfer, lower) = (timed(approach)?, timed(descend)?, timed(transfer)?, timed(lower)?);
        let mut episode = Episode::new(n, o.dt);
        episode.append(&approach, 1.0);
        episode.append(&descend, 1.0);
        let grasped = episode.len();
        episode.hold(hold, 1.0, 0.0);
        episode.append(&reversed(&descend), 0.0);
        episode.append(&transfer, 0.0);
        episode.append(&lower, 0.0);
        let released = episode.len();
        episode.hold(hold, 0.0, 1.0);
        episode.append(&reversed(&lower), 1.0);

        let g = &grasps[i];
        let carried = |q: &[f32]| robot.ee_pose(q).mul_pose(g.object_in_tool);
        let last_held = carried(&episode.trajectory.positions[(released - 1) * n..released * n]);
        let poses = (0..episode.len())
            .map(|f| match f {
                f if f < grasped + hold => g.object,
                f if f < released => carried(&episode.trajectory.positions[f * n..(f + 1) * n]),
                _ => last_held,
            })
            .collect();
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: tasks[i].world,
            task: describe(g.object.position, tasks[i].place.position),
            goal: tasks[i].place,
            trajectory: episode.trajectory,
            gripper: episode.gripper,
            carried: Some(Carried { obstacle: tasks[i].object, poses }),
        });
    }
    Ok(demos)
}

/// A top-down grasp of an upright cuboid.
struct Grasp {
    object: Pose,
    half: Vec3,
    /// The IK frame at the grasp.
    tool: Pose,
    /// The object's pose in the IK frame while held.
    object_in_tool: Pose,
}

impl Grasp {
    fn new(object: Pose, half: Vec3, depth: f32) -> Self {
        // Fingers close across the narrower side; the hand faces away from the robot's base.
        let z = -Vec3::Z;
        let mut y = object.rotation * if half.x <= half.y { Vec3::X } else { Vec3::Y };
        if y.cross(z).dot(object.position.with_z(0.0)) < 0.0 {
            y = -y;
        }
        let rotation = Quat::from_mat3(&glam::Mat3::from_cols(y.cross(z), y, z));
        let position = object.position + Vec3::Z * (half.z - depth.min(half.z));
        let tool = Pose { position, rotation };
        Self { object, half, tool, object_in_tool: tool.inverse().mul_pose(object) }
    }

    /// The IK frame that holds the object at `place`.
    fn tool_for(&self, place: Pose) -> Pose {
        place.mul_pose(self.object_in_tool.inverse())
    }
}

/// "Pick up the box and put it 20 cm to the left.": the move along its larger horizontal axis,
/// in the robot base's frame (x forward, y left), to the nearest 5 cm.
fn describe(from: Vec3, to: Vec3) -> String {
    let d = to - from;
    let (along, direction) = if d.x.abs() >= d.y.abs() {
        (d.x, if d.x >= 0.0 { "forward" } else { "back" })
    } else {
        (d.y, if d.y >= 0.0 { "to the left" } else { "to the right" })
    };
    let cm = ((along.abs() * 20.0).round() * 5.0).max(5.0);
    format!("Pick up the box and put it {cm} cm {direction}.")
}

/// A demonstration being assembled from timed segments that start and end at rest.
struct Episode {
    trajectory: JointTrajectory,
    gripper: Vec<f32>,
}

impl Episode {
    fn new(dof: usize, dt: f32) -> Self {
        let trajectory =
            JointTrajectory { dof, dt, duration: 0.0, positions: vec![], velocities: vec![], accelerations: vec![] };
        Self { trajectory, gripper: vec![] }
    }

    fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Appends `segment` with the gripper at `opening`; its first sample, at rest where the
    /// episode ends, is dropped after the first segment.
    fn append(&mut self, segment: &JointTrajectory, opening: f32) {
        let n = self.trajectory.dof;
        let skip = usize::from(!self.trajectory.positions.is_empty());
        let t = &mut self.trajectory;
        t.positions.extend_from_slice(&segment.positions[skip * n..]);
        t.velocities.extend_from_slice(&segment.velocities[skip * n..]);
        t.accelerations.extend_from_slice(&segment.accelerations[skip * n..]);
        self.gripper.resize(t.positions.len() / n, opening);
        t.duration = (self.gripper.len() - 1) as f32 * t.dt;
    }

    /// Holds the arm still for `samples` samples while the gripper moves from `from` to `to`.
    fn hold(&mut self, samples: usize, from: f32, to: f32) {
        let n = self.trajectory.dof;
        let t = &mut self.trajectory;
        let last = t.positions[t.positions.len() - n..].to_vec();
        for k in 1..=samples {
            t.positions.extend_from_slice(&last);
            t.velocities.extend(std::iter::repeat_n(0.0, n));
            t.accelerations.extend(std::iter::repeat_n(0.0, n));
            self.gripper.push(from + (to - from) * k as f32 / samples as f32);
        }
        t.duration = (self.gripper.len() - 1) as f32 * t.dt;
    }
}

/// `t` played backwards.
fn reversed(t: &JointTrajectory) -> JointTrajectory {
    let n = t.dof;
    let rows = |v: &[f32], sign: f32| v.chunks(n).rev().flat_map(|r| r.iter().map(move |x| sign * x)).collect();
    JointTrajectory {
        positions: rows(&t.positions, 1.0),
        velocities: rows(&t.velocities, -1.0),
        accelerations: rows(&t.accelerations, 1.0),
        ..t.clone()
    }
}
