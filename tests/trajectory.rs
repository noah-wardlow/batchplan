//! Executable trajectories: limits hold everywhere along planned paths, sampling a trajectory does
//! not allocate, and `check` refuses unsafe trajectories.

#[path = "../examples/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use batchplan::rng::Rng;
use batchplan::*;

/// Counts allocations made by the current thread, so other test threads cannot interfere.
struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Collision-free plans from the default pose to IK solutions in tabletop worlds.
fn planned(robot: &Robot, count: usize) -> Vec<Vec<f32>> {
    let cpu = Device::cpu(robot);
    let mut rng = Rng::new(31);
    let worlds: Vec<World> = (0..count).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, target: common::grasp_target(w, &mut rng) })
        .collect();
    let worlds = cpu.upload(&worlds).unwrap();
    let ik = solve_ik(&cpu, &worlds, &goals, &IkOptions::default()).unwrap();
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec() })
        .collect();
    let result = plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let paths: Vec<Vec<f32>> = result.solved().map(|s| s.solution.to_vec()).collect();
    assert!(paths.len() >= count * 3 / 4, "only {} of {count} problems planned", paths.len());
    paths
}

#[test]
fn trajectories_stay_within_limits_everywhere() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let limits = [robot.max_velocity(), robot.max_acceleration(), robot.max_jerk()];
    let mut peak = [0.0f32; 3];
    for path in planned(&robot, 16) {
        let trajectory = Trajectory::new(&robot, &path, 1.0);
        trajectory.check(&robot).unwrap();
        let dt = trajectory.knot_interval / 64.0;
        let samples = trajectory.sample(1.0 / dt);
        let (p, v, a) = (&samples.positions, &samples.velocities, &samples.accelerations);
        assert_eq!(&p[..n], &path[..n], "starts exactly at the start");
        assert_eq!(&p[p.len() - n..], &path[path.len() - n..], "ends exactly at the goal");
        assert!(v[..n].iter().chain(&v[v.len() - n..]).all(|&x| x == 0.0), "at rest at both ends");
        for i in 0..p.len() {
            let j = i % n;
            // The curve stays in the control points' hull, up to f32 rounding of the blend.
            let excess = (robot.lower()[j] - p[i]).max(p[i] - robot.upper()[j]);
            assert!(excess < 1e-5, "joint {j} leaves its range by {excess}");
            peak[0] = peak[0].max(v[i].abs() / limits[0][j]);
            peak[1] = peak[1].max(a[i].abs() / limits[1][j]);
        }
        // Jerk is constant along each span, so finite differences of acceleration never exceed it.
        // A coarser step keeps f32 cancellation in the accelerations from dominating.
        let coarse_dt = trajectory.knot_interval / 4.0;
        let coarse = trajectory.sample(1.0 / coarse_dt).accelerations;
        for i in n..coarse.len() - n {
            peak[2] = peak[2].max(((coarse[i] - coarse[i - n]) / coarse_dt).abs() / limits[2][i % n]);
        }
        // The analytic derivatives match finite differences of the samples (away from the ends,
        // where sampling clamps).
        for i in 2 * n..p.len() - 2 * n {
            let fd_v = (p[i + n] - p[i - n]) / (2.0 * dt);
            let fd_a = (v[i + n] - v[i - n]) / (2.0 * dt);
            assert!((fd_v - v[i]).abs() < 2e-3 * limits[0][i % n], "velocity off: {} vs {fd_v}", v[i]);
            assert!((fd_a - a[i]).abs() < 2e-2 * limits[1][i % n], "acceleration off: {} vs {fd_a}", a[i]);
        }
    }
    eprintln!("peak fraction of the velocity, acceleration and jerk limits: {peak:?}");
    assert!(peak.iter().all(|&p| p <= 1.0 + 1e-3), "a limit is exceeded: {peak:?}");
    // The binding limit is reached: the timing is as fast as the limits allow.
    assert!(peak.iter().any(|&p| p > 0.9), "no limit is close to binding: {peak:?}");
}

#[test]
fn slower_speed_scales_stretch_time() {
    let robot = common::panda().unwrap();
    let path = &planned(&robot, 4)[0];
    let fast = Trajectory::new(&robot, path, 1.0);
    let slow = Trajectory::new(&robot, path, 0.5);
    assert!((slow.duration() - 2.0 * fast.duration()).abs() < 1e-5 * fast.duration());
    slow.check(&robot).unwrap();
}

#[test]
fn sampling_does_not_allocate() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::new(&robot, &planned(&robot, 4)[0], 1.0);
    let mut state = JointState::new(robot.dof());
    let before = ALLOCATIONS.with(Cell::get);
    let mut checksum = 0.0;
    for k in 0..=1000 {
        trajectory.at(trajectory.duration() * k as f32 / 1000.0, &mut state);
        checksum += state.position[0] + state.velocity[0] + state.acceleration[0];
    }
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    assert!(checksum.is_finite());
    assert_eq!(allocations, 0, "Trajectory::at allocated {allocations} times");
    // The counter does see allocations on this thread.
    let probe = std::hint::black_box(Box::new(1u8));
    assert_eq!(ALLOCATIONS.with(Cell::get) - before, 1);
    drop(probe);
}

#[test]
fn check_rejects_unsafe_trajectories() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let good = Trajectory::new(&robot, &planned(&robot, 4)[0], 1.0);
    good.check(&robot).unwrap();
    let points = good.control_points.len() / n;
    let rejects = |t: Trajectory, why: &str| {
        let err = t.check(&robot).expect_err(why).to_string();
        eprintln!("{why}: {err}");
    };

    let mut nan = good.clone();
    nan.control_points[5 * n + 2] = f32::NAN;
    rejects(nan, "a NaN");

    rejects(Trajectory { knot_interval: good.knot_interval * 0.8, ..good.clone() }, "running too fast");

    // A long move over few control points, where velocity is the binding limit, sped up 10%.
    let mut cp = vec![];
    for q0 in [-2.5, -2.5, -2.5, 0.0, 2.5, 2.5, 2.5] {
        let mut q = robot.default_q().to_vec();
        q[0] = q0;
        cp.extend(q);
    }
    let sweep = Trajectory::new(&robot, &cp, 1.0);
    sweep.check(&robot).unwrap();
    let err = Trajectory { knot_interval: sweep.knot_interval * 0.9, ..sweep }.check(&robot).unwrap_err().to_string();
    assert!(err.contains("velocity"), "{err}");

    let mut spike = good.clone();
    spike.control_points[(points / 2) * n] += 0.05;
    let err = spike.check(&robot).expect_err("a jerk spike").to_string();
    assert!(err.contains("jerk") || err.contains("acceleration"), "{err}");

    let mut moving_start = good.clone();
    moving_start.control_points[n] += 0.01;
    rejects(moving_start, "not at rest at the start");

    let mut out_of_range = good.clone();
    out_of_range.control_points[(points / 2) * n + 3] = robot.upper()[3] + 0.1;
    rejects(out_of_range, "leaving a joint range");

    rejects(Trajectory { dof: n - 1, ..good.clone() }, "the wrong number of joints");
}

#[test]
fn trajectories_round_trip_through_json() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::new(&robot, &planned(&robot, 4)[0], 0.7);
    let text = serde_json::to_string(&trajectory).unwrap();
    let back: Trajectory = serde_json::from_str(&text).unwrap();
    assert_eq!(back, trajectory);
    back.check(&robot).unwrap();
}
