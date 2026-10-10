//! RRT-Connect (Kuffner and LaValle, 2000) with goal sets, for problems trajectory optimization
//! leaves unsolved. Problems advance in lockstep. Each round extends every unsolved problem's
//! tree toward several random configurations at once and connects its other tree toward each new
//! node; each of those two steps checks all problems' edges in one batched [`Device::evaluate`]
//! call, so a device sees few, large batches.

use std::time::Instant;

use crate::error::{Result, ensure_input};

use crate::device::{CollisionWeights, Device, Worlds};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct RrtOptions {
    /// Longest edge between tree nodes (joint-space distance).
    pub step: f32,
    /// Spacing of collision checks along edges (joint-space distance).
    pub resolution: f32,
    /// Rounds before an unsolved problem gives up.
    pub max_rounds: usize,
    /// Random configurations each round extends toward.
    pub extensions_per_round: usize,
    pub rng_seed: u64,
}

impl Default for RrtOptions {
    fn default() -> Self {
        Self { step: 0.5, resolution: 0.02, max_rounds: 150, extensions_per_round: 8, rng_seed: 3 }
    }
}

#[derive(Clone, Debug)]
pub struct RrtProblem {
    pub world: u32,
    pub start: Vec<f32>,
    /// The path may end at any of these.
    pub goals: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct RrtResult {
    pub problems: Vec<RrtProblem>,
    /// Each problem's path as waypoints (`[k, dof]`, start first), if one was found. Consecutive
    /// waypoints are at most `step` apart and the edges between them are collision-free at
    /// `resolution`.
    pub paths: Vec<Option<Vec<f32>>>,
}

/// Straight joint-space segments, checked together in one batched evaluation.
pub(crate) struct Segments {
    resolution: f32,
    dof: usize,
    samples: Vec<f32>,
    item_world: Vec<u32>,
    /// Each segment's samples, `start..end` in `item_world`.
    spans: Vec<(usize, usize)>,
}

impl Segments {
    pub(crate) fn new(dof: usize, resolution: f32) -> Self {
        Self { resolution, dof, samples: vec![], item_world: vec![], spans: vec![] }
    }

    /// Samples `a` to `b` at `resolution`, excluding `a` and including `b`.
    pub(crate) fn push(&mut self, world: u32, a: &[f32], b: &[f32]) {
        let count = (distance(a, b) / self.resolution).ceil().max(1.0) as usize;
        let first = self.item_world.len();
        for k in 1..=count {
            let t = k as f32 / count as f32;
            self.samples.extend(a.iter().zip(b).map(|(x, y)| x + (y - x) * t));
            self.item_world.push(world);
        }
        self.spans.push((first, first + count));
    }

    /// For each segment, the fraction of it known free: 1 when every sample is collision-free,
    /// otherwise up to the sample before the first collision.
    pub(crate) fn check(&self, device: &Device, worlds: &Worlds) -> Result<Vec<f32>> {
        debug_assert_eq!(self.samples.len(), self.item_world.len() * self.dof);
        let eval = device.evaluate(worlds, &self.item_world, &self.samples, &CollisionWeights::NONE)?;
        Ok(self
            .spans
            .iter()
            .map(|&(first, end)| match (first..end).position(|i| !eval.collision_free(i)) {
                Some(k) => k as f32 / (end - first) as f32,
                None => 1.0,
            })
            .collect())
    }
}

pub(crate) fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// Nodes with parents; roots have none.
struct Tree {
    dof: usize,
    nodes: Vec<f32>,
    parent: Vec<Option<usize>>,
}

impl Tree {
    fn new(dof: usize) -> Self {
        Self { dof, nodes: vec![], parent: vec![] }
    }

    fn node(&self, i: usize) -> &[f32] {
        &self.nodes[i * self.dof..(i + 1) * self.dof]
    }

    fn add(&mut self, q: &[f32], parent: Option<usize>) -> usize {
        self.nodes.extend_from_slice(q);
        self.parent.push(parent);
        self.parent.len() - 1
    }

    fn nearest(&self, q: &[f32]) -> usize {
        let squared = |i: usize| self.node(i).iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        (0..self.parent.len()).map(|i| (squared(i), i)).min_by(|a, b| a.0.total_cmp(&b.0)).expect("trees have a root").1
    }

    /// Node `i` and its ancestors up to the root, `i` first.
    fn branch(&self, mut i: usize) -> Vec<f32> {
        let mut out = self.node(i).to_vec();
        while let Some(p) = self.parent[i] {
            out.extend_from_slice(self.node(p));
            i = p;
        }
        out
    }
}

struct Search {
    world: u32,
    /// The tree that extends next, then the one that connects; `start_first` while the first
    /// grows from the start.
    trees: [Tree; 2],
    start_first: bool,
    rng: Rng,
    path: Option<Vec<f32>>,
}

impl Search {
    fn swap(&mut self) {
        self.trees.swap(0, 1);
        self.start_first = !self.start_first;
    }
}

/// Finds a collision-free waypoint path from each problem's start to one of its goals. Starts and
/// goals in collision are skipped; a problem without a free start or goal has no path.
pub fn connect(device: &Device, worlds: &Worlds, problems: &[RrtProblem], o: &RrtOptions) -> Result<RrtResult> {
    connect_until(device, worlds, problems, o, None)
}

/// [`connect`] that stops searching between rounds once `deadline` passes.
pub(crate) fn connect_until(
    device: &Device,
    worlds: &Worlds,
    problems: &[RrtProblem],
    o: &RrtOptions,
    deadline: Option<Instant>,
) -> Result<RrtResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(o.step > 0.0 && o.resolution > 0.0, "step and resolution must be positive");
    ensure_input!(o.extensions_per_round > 0, "extend toward at least one configuration per round");
    let within = |q: &[f32]| q.iter().enumerate().all(|(j, &v)| v >= robot.lower[j] && v <= robot.upper[j]);
    let mut ends = vec![];
    let mut end_world = vec![];
    for (i, p) in problems.iter().enumerate() {
        ensure_input!(p.start.len() == n, "problem {i}: the start must have {n} values");
        ensure_input!(p.goals.iter().all(|g| g.len() == n), "problem {i}: goals must have {n} values");
        ensure_input!(
            within(&p.start) && p.goals.iter().all(|g| within(g)),
            "problem {i}: the start and goals must be within the joint limits"
        );
        for q in std::iter::once(&p.start).chain(&p.goals) {
            ends.extend_from_slice(q);
            end_world.push(p.world);
        }
    }
    let free = device.evaluate(worlds, &end_world, &ends, &CollisionWeights::NONE)?;
    let mut searches = vec![];
    let mut next_end = 0;
    for (i, p) in problems.iter().enumerate() {
        let mut start = Tree::new(n);
        if free.collision_free(next_end) {
            start.add(&p.start, None);
        }
        let mut goals = Tree::new(n);
        for (k, g) in p.goals.iter().enumerate() {
            if free.collision_free(next_end + 1 + k) {
                goals.add(g, None);
            }
        }
        next_end += 1 + p.goals.len();
        let rng = Rng::new(o.rng_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        searches.push(Search { world: p.world, trees: [start, goals], start_first: true, rng, path: None });
    }
    let alive = |s: &Search| s.path.is_none() && s.trees.iter().all(|t| !t.parent.is_empty());

    for _ in 0..o.max_rounds {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // Extend: a step from the nearest node toward each of several random configurations.
        let mut extend = Segments::new(n, o.resolution);
        let mut steps = vec![];
        let active: Vec<usize> = (0..searches.len()).filter(|&s| alive(&searches[s])).collect();
        if active.is_empty() {
            break;
        }
        for &s in &active {
            let search = &mut searches[s];
            for _ in 0..o.extensions_per_round {
                let target: Vec<f32> = (0..n).map(|j| search.rng.range(robot.lower[j], robot.upper[j])).collect();
                let tree = &search.trees[0];
                let near = tree.nearest(&target);
                let from = tree.node(near);
                let reach = (o.step / distance(from, &target)).min(1.0);
                let new: Vec<f32> = from.iter().zip(&target).map(|(a, b)| a + (b - a) * reach).collect();
                extend.push(search.world, from, &new);
                steps.push((s, near, new));
            }
        }
        // Connect: the other tree steps toward each new node until it arrives or is blocked.
        let mut connect = Segments::new(n, o.resolution);
        let mut joins = vec![];
        for ((s, near, new), known) in steps.into_iter().zip(extend.check(device, worlds)?) {
            if known < 1.0 {
                continue;
            }
            let search = &mut searches[s];
            let added = search.trees[0].add(&new, Some(near));
            let other = search.trees[1].nearest(&new);
            connect.push(search.world, search.trees[1].node(other), &new);
            joins.push((s, added, other));
        }
        for ((s, added, other), known) in joins.into_iter().zip(connect.check(device, worlds)?) {
            let search = &mut searches[s];
            if search.path.is_some() {
                continue;
            }
            let [extended, connecting] = &mut search.trees;
            let (from, to) = (connecting.node(other).to_vec(), extended.node(added).to_vec());
            let length = distance(&from, &to);
            let mut parent = other;
            let mut along = o.step;
            while along < length && along <= known * length {
                let q: Vec<f32> = from.iter().zip(&to).map(|(a, b)| a + (b - a) * (along / length)).collect();
                parent = connecting.add(&q, Some(parent));
                along += o.step;
            }
            if known == 1.0 {
                let mut path: Vec<f32> = extended.branch(added).chunks(n).rev().flatten().copied().collect();
                path.extend(connecting.branch(parent));
                if !search.start_first {
                    path = path.chunks(n).rev().flatten().copied().collect();
                }
                search.path = Some(path);
            }
        }
        for s in active {
            searches[s].swap();
        }
    }
    Ok(RrtResult { problems: problems.to_vec(), paths: searches.into_iter().map(|s| s.path).collect() })
}
