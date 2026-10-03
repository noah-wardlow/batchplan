//! CPU device checks: kinematics against published parameters, analytic gradients, IK, retiming.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f64::consts::FRAC_PI_2;

use batchplan::rng::Rng;
use batchplan::timing::{RetimeOptions, retime};
use batchplan::*;
use glam::{DMat3, DVec3};

fn panda() -> Robot {
    Robot::from_config_file(common::panda_config()).unwrap()
}

fn random_q(robot: &Robot, rng: &mut Rng) -> Vec<f32> {
    (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect()
}

/// Flange pose from Franka's published modified-DH table, independent of the URDF parser.
fn franka_dh_flange(q: &[f32]) -> (DMat3, DVec3) {
    let table = [
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -FRAC_PI_2),
        (0.0, 0.316, FRAC_PI_2),
        (0.0825, 0.0, FRAC_PI_2),
        (-0.0825, 0.384, -FRAC_PI_2),
        (0.0, 0.0, FRAC_PI_2),
        (0.088, 0.0, FRAC_PI_2),
    ];
    let (mut r, mut p) = (DMat3::IDENTITY, DVec3::ZERO);
    for (i, &(a, d, alpha)) in table.iter().enumerate() {
        // T_i = RotX(alpha) * TransX(a) * RotZ(q) * TransZ(d)
        p += r * DVec3::new(a, 0.0, 0.0);
        r *= DMat3::from_rotation_x(alpha) * DMat3::from_rotation_z(q[i] as f64);
        p += r * DVec3::new(0.0, 0.0, d);
    }
    p += r * DVec3::new(0.0, 0.0, 0.107);
    (r, p)
}

#[test]
fn fk_matches_franka_dh_parameters() {
    let robot = panda();
    assert_eq!(robot.dof(), 7);
    let mut rng = Rng::new(7);
    for _ in 0..200 {
        let q = random_q(&robot, &mut rng);
        let pose = robot.link_pose(&q, "panda_link8").unwrap();
        let (r, p) = franka_dh_flange(&q);
        assert!((pose.position.as_dvec3() - p).length() < 1e-5, "position mismatch at {q:?}");
        let r_err =
            (DMat3::from_quat(pose.rotation.as_dquat()) - r).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(r_err < 1e-5, "rotation mismatch {r_err} at {q:?}");
    }
}

#[test]
fn default_pose_is_collision_free_on_table() {
    let robot = panda();
    let cpu = Device::cpu(&robot);
    let world = common::tabletop(&mut Rng::new(1));
    let table_only = World { obstacles: world.obstacles[..1].to_vec() };
    let e = cpu.evaluate(&[table_only], &[0], robot.default_q(), &CollisionWeights::NONE).unwrap();
    assert!(e.world_clearance[0] > 0.05, "world clearance {}", e.world_clearance[0]);
    assert!(e.self_clearance[0] > 0.0, "self clearance {}", e.self_clearance[0]);
}

#[test]
fn collision_gradient_matches_finite_differences() {
    let robot = panda();
    let cpu = Device::cpu(&robot);
    let n = robot.dof();
    let w = CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.05, self_margin: 0.02 };
    let mut rng = Rng::new(11);
    let worlds: Vec<World> = (0..8).map(|_| common::tabletop(&mut rng)).collect();
    let mut checked = 0;
    for trial in 0..400 {
        let q = random_q(&robot, &mut rng);
        let world = trial as u32 % 8;
        let e = cpu.evaluate(&worlds, &[world], &q, &w).unwrap();
        if e.cost[0] < 1e-3 {
            continue;
        }
        let g = &e.grad;
        let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        // The cost is only piecewise smooth (box faces, hinge activation), so a finite difference
        // straddling a kink is wrong at that step size; accept agreement at either step size.
        let rel_err = |h: f32| {
            let fd: Vec<f32> = (0..n)
                .map(|j| {
                    let (mut qp, mut qm) = (q.clone(), q.clone());
                    qp[j] += h;
                    qm[j] -= h;
                    let cp = cpu.evaluate(&worlds, &[world], &qp, &w).unwrap().cost[0];
                    let cm = cpu.evaluate(&worlds, &[world], &qm, &w).unwrap().cost[0];
                    (cp - cm) / (2.0 * h)
                })
                .collect();
            (0..n).map(|j| (fd[j] - g[j]).powi(2)).sum::<f32>().sqrt() / norm
        };
        let err = rel_err(2e-4).min(rel_err(5e-5));
        assert!(err < 2e-2, "gradient mismatch {err} (|g| = {norm}) at {q:?}, grad {g:?}");
        checked += 1;
    }
    assert!(checked > 50, "only {checked} colliding samples");
}

#[test]
fn ik_reaches_targets_from_collision_free_configurations() {
    let robot = panda();
    let cpu = Device::cpu(&robot);
    let mut rng = Rng::new(5);
    let mut problems = vec![];
    while problems.len() < 64 {
        let q = random_q(&robot, &mut rng);
        if cpu.evaluate(&[World::default()], &[0], &q, &CollisionWeights::NONE).unwrap().collision_free(0) {
            problems.push(IkProblem { world: 0, target: robot.ee_pose(&q) });
        }
    }
    let o = IkOptions { seeds: 16, ..Default::default() };
    let result = solve_ik(&cpu, &[World::default()], &problems, &o).unwrap();
    let solved = (0..problems.len()).filter(|&p| result.best(p).is_some()).count();
    assert!(solved >= 60, "solved {solved}/64");
    for (p, problem) in problems.iter().enumerate() {
        if let Some(q) = result.best(p) {
            assert!((robot.ee_pose(q).position - problem.target.position).length() < o.position_tolerance);
        }
    }
}

#[test]
fn retime_keeps_endpoints_and_velocity_limits() {
    let robot = panda();
    let path: Vec<f32> = (0..10).flat_map(|t| robot.default_q().iter().map(move |q| q + 0.1 * t as f32)).collect();
    let tt = retime(&robot, &path, &RetimeOptions { max_acceleration: 10.0, speed_scale: 1.0, dt: 0.01 });
    assert!(tt.duration > 0.0);
    let (first, last) = (&tt.positions[..7], &tt.positions[tt.positions.len() - 7..]);
    assert!(first.iter().zip(&path[..7]).all(|(a, b)| (a - b).abs() < 1e-6));
    assert!(last.iter().zip(&path[63..]).all(|(a, b)| (a - b).abs() < 1e-5));
    for row in tt.velocities.chunks(7) {
        assert!(row.iter().zip(robot.max_velocity()).all(|(v, max)| v.abs() <= max * 1.001));
    }
}

#[test]
fn npy_files_have_aligned_headers() {
    let path = std::env::temp_dir().join("batchplan-npy-test.npy");
    batchplan::npy::write_npy(&path, &[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0);
    assert_eq!(bytes.len(), 10 + header_len + 24);
}
