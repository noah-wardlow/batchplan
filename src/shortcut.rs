//! Shortening waypoint paths: random shortcuts until a run of attempts fails, then removing
//! waypoints whose neighbors see each other. Every candidate edge is collision-checked. Paths
//! advance in lockstep, each round trying several shortcuts per path and keeping the one that
//! shortens it most, so each round checks all of them in one batched evaluation.

use std::time::Instant;

use crate::error::{Result, ensure_input};

use crate::device::{Device, Worlds};
use crate::rng::Rng;
use crate::rrt::{Segments, distance};

#[derive(Clone, Copy, Debug)]
pub struct ShortcutOptions {
    /// Spacing of collision checks along new edges (joint-space distance).
    pub resolution: f32,
    /// A path is done after this many failed shortcuts in a row.
    pub patience: usize,
    /// Shortcut attempts per path at most.
    pub max_attempts: usize,
    /// Shortcuts each round tries per path.
    pub attempts_per_round: usize,
    pub rng_seed: u64,
}

impl Default for ShortcutOptions {
    fn default() -> Self {
        Self { resolution: 0.02, patience: 32, max_attempts: 320, attempts_per_round: 8, rng_seed: 5 }
    }
}

/// Shortens each path (`[k, dof]` waypoints in world `world[i]`) in place. Endpoints stay; the
/// length never grows.
pub fn shortcut(
    device: &Device,
    worlds: &Worlds,
    world: &[u32],
    paths: &mut [Vec<f32>],
    o: &ShortcutOptions,
) -> Result<()> {
    shortcut_until(device, worlds, world, paths, o, None)
}

/// [`shortcut`] that stops between rounds once `deadline` passes.
pub(crate) fn shortcut_until(
    device: &Device,
    worlds: &Worlds,
    world: &[u32],
    paths: &mut [Vec<f32>],
    o: &ShortcutOptions,
    deadline: Option<Instant>,
) -> Result<()> {
    let n = device.robot().dof();
    let expired = || deadline.is_some_and(|d| Instant::now() >= d);
    ensure_input!(o.attempts_per_round > 0, "try at least one shortcut per round");
    ensure_input!(world.len() == paths.len(), "{} worlds for {} paths", world.len(), paths.len());
    ensure_input!(paths.iter().all(|p| !p.is_empty() && p.len() % n == 0), "paths must hold whole waypoints");
    let mut rngs: Vec<Rng> =
        (0..paths.len()).map(|i| Rng::new(o.rng_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))).collect();
    let (mut failures, mut attempts) = (vec![0; paths.len()], vec![0; paths.len()]);
    let active = |failures: usize, attempts: usize, path: &[f32]| {
        failures < o.patience && attempts < o.max_attempts && path.len() >= 3 * n
    };
    while !expired() && (0..paths.len()).any(|i| active(failures[i], attempts[i], &paths[i])) {
        let mut segments = Segments::new(n, o.resolution);
        let mut tries = vec![];
        let mut round = vec![];
        for (i, path) in paths.iter().enumerate() {
            if !active(failures[i], attempts[i], path) {
                continue;
            }
            round.push(i);
            let total = length(path, n);
            for _ in 0..o.attempts_per_round {
                let (a, b) = (rngs[i].uniform() * total, rngs[i].uniform() * total);
                let ((ia, pa), (ib, pb)) = (locate(path, n, a.min(b)), locate(path, n, a.max(b)));
                if ia != ib {
                    segments.push(world[i], &pa, &pb);
                    tries.push((i, ia, pa, ib, pb));
                }
            }
        }
        // Each path keeps its shortest result.
        let mut best: Vec<Option<Vec<f32>>> = vec![None; paths.len()];
        for ((i, ia, pa, ib, pb), known) in tries.into_iter().zip(segments.check(device, worlds)?) {
            // Waypoints 0..=ia, then the shortcut, then ib + 1.. (pa lies on edge ia, pb on edge ib).
            let path = &paths[i];
            let mut shorter = path[..(ia + 1) * n].to_vec();
            shorter.extend(pa.iter().chain(&pb));
            shorter.extend_from_slice(&path[(ib + 1) * n..]);
            let bar = best[i].as_deref().map_or(length(path, n), |b| length(b, n));
            if known == 1.0 && length(&shorter, n) < bar {
                best[i] = Some(shorter);
            }
        }
        for i in round {
            attempts[i] += o.attempts_per_round;
            match best[i].take() {
                Some(shorter) => {
                    paths[i] = shorter;
                    failures[i] = 0;
                }
                None => failures[i] += o.attempts_per_round,
            }
        }
    }
    // Drop waypoints whose neighbors are connected by a free edge.
    let mut next = vec![1; paths.len()];
    while !expired() {
        let mut segments = Segments::new(n, o.resolution);
        let mut tries = vec![];
        for (i, path) in paths.iter().enumerate() {
            let k = next[i];
            if (k + 1) * n < path.len() {
                segments.push(world[i], &path[(k - 1) * n..k * n], &path[(k + 1) * n..(k + 2) * n]);
                tries.push(i);
            }
        }
        if tries.is_empty() {
            break;
        }
        for (i, known) in tries.into_iter().zip(segments.check(device, worlds)?) {
            if known == 1.0 {
                paths[i].drain(next[i] * n..(next[i] + 1) * n);
            } else {
                next[i] += 1;
            }
        }
    }
    Ok(())
}

pub(crate) fn length(path: &[f32], n: usize) -> f32 {
    path.chunks(n).zip(path.chunks(n).skip(1)).map(|(a, b)| distance(a, b)).sum()
}

/// The point `along` the path (by joint-space distance) and the edge it lies on.
pub(crate) fn locate(path: &[f32], n: usize, along: f32) -> (usize, Vec<f32>) {
    let mut walked = 0.0;
    let edges = path.len() / n - 1;
    for e in 0..edges {
        let (a, b) = (&path[e * n..(e + 1) * n], &path[(e + 1) * n..(e + 2) * n]);
        let d = distance(a, b);
        if walked + d >= along || e + 1 == edges {
            let t = if d > 0.0 { ((along - walked) / d).clamp(0.0, 1.0) } else { 0.0 };
            return (e, a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect());
        }
        walked += d;
    }
    unreachable!("paths have at least one edge")
}
