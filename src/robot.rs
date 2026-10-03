//! Robot model: a kinematic tree loaded from URDF plus a collision-sphere approximation.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{Mat3, Quat, Vec3};
use serde::Deserialize;

use crate::types::Pose;

/// Limits shared with the GPU kernels (private per-invocation arrays are sized from these).
pub(crate) const MAX_DOF: usize = 16;
pub(crate) const MAX_LINKS: usize = 32;
pub(crate) const MAX_SPHERES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Transform {
    pub(crate) rot: Mat3,
    pub(crate) trans: Vec3,
}

impl Transform {
    pub(crate) const IDENTITY: Self = Self { rot: Mat3::IDENTITY, trans: Vec3::ZERO };

    /// URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    pub(crate) fn from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Self {
        let [r, p, y] = rpy.map(|v| v as f32);
        Self {
            rot: Mat3::from_rotation_z(y) * Mat3::from_rotation_y(p) * Mat3::from_rotation_x(r),
            trans: Vec3::from_array(xyz.map(|v| v as f32)),
        }
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        Self { rot: self.rot * other.rot, trans: self.rot * other.trans + self.trans }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JointKind {
    Fixed,
    Revolute { dof: usize, axis: Vec3 },
    Prismatic { dof: usize, axis: Vec3 },
}

#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) name: String,
    /// Parent link index; parents always precede children. `None` for the root.
    pub(crate) parent: Option<usize>,
    /// Joint frame relative to the parent link frame (the URDF joint origin).
    pub(crate) origin: Transform,
    pub(crate) joint: JointKind,
    /// Bitmask of the actuated joints that move this link.
    pub(crate) dof_mask: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CollisionSphere {
    pub(crate) link: usize,
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
    /// Extra radius used only for self-collision.
    pub(crate) self_buffer: f32,
}

/// A robot's kinematic tree and collision-sphere model. Immutable once loaded.
#[derive(Clone, Debug)]
pub struct Robot {
    pub(crate) name: String,
    pub(crate) links: Vec<Link>,
    pub(crate) dof_names: Vec<String>,
    pub(crate) lower: Vec<f32>,
    pub(crate) upper: Vec<f32>,
    pub(crate) max_velocity: Vec<f32>,
    pub(crate) spheres: Vec<CollisionSphere>,
    /// Sphere index pairs checked for self-collision.
    pub(crate) self_pairs: Vec<[u32; 2]>,
    /// Link whose frame is the IK target frame.
    pub(crate) ee_link: usize,
    pub(crate) default_q: Vec<f32>,
}

#[derive(Deserialize)]
struct RobotConfig {
    urdf: String,
    base_link: String,
    ee_link: String,
    #[serde(default)]
    lock_joints: HashMap<String, f64>,
    default_q: Vec<f32>,
    /// Per link: `[x, y, z, radius]` in the link frame.
    spheres: HashMap<String, Vec<[f32; 4]>>,
    #[serde(default)]
    self_collision_buffer: HashMap<String, f32>,
    #[serde(default)]
    self_collision_ignore: HashMap<String, Vec<String>>,
}

/// World-frame kinematic state for one configuration.
#[derive(Clone, Copy)]
pub(crate) struct Fk {
    pub(crate) rot: [Mat3; MAX_LINKS],
    pub(crate) pos: [Vec3; MAX_LINKS],
    /// World axis and anchor point of every actuated joint.
    pub(crate) axis: [Vec3; MAX_DOF],
    pub(crate) anchor: [Vec3; MAX_DOF],
    pub(crate) prismatic: u32,
}

impl Fk {
    /// Derivative of a point rigidly attached downstream of joint `dof`.
    pub(crate) fn dpoint(&self, dof: usize, p: Vec3) -> Vec3 {
        if self.prismatic >> dof & 1 == 1 { self.axis[dof] } else { self.axis[dof].cross(p - self.anchor[dof]) }
    }
}

impl Robot {
    /// Loads a robot from a JSON config that points at a URDF (path relative to the config).
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: RobotConfig = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let urdf_path = path.parent().unwrap_or(Path::new(".")).join(&cfg.urdf);
        let urdf = urdf_rs::read_file(&urdf_path).with_context(|| format!("parsing {}", urdf_path.display()))?;
        Self::build(&urdf, &cfg)
    }

    fn build(urdf: &urdf_rs::Robot, cfg: &RobotConfig) -> Result<Self> {
        use urdf_rs::JointType as J;
        let mut links = vec![Link {
            name: cfg.base_link.clone(),
            parent: None,
            origin: Transform::IDENTITY,
            joint: JointKind::Fixed,
            dof_mask: 0,
        }];
        let (mut dof_names, mut lower, mut upper, mut max_velocity) = (vec![], vec![], vec![], vec![]);
        // Breadth-first over the URDF tree so parents precede children.
        let mut i = 0;
        while i < links.len() {
            let (parent_name, parent_mask) = (links[i].name.clone(), links[i].dof_mask);
            for j in urdf.joints.iter().filter(|j| j.parent.link == parent_name) {
                let mut origin = Transform::from_xyz_rpy(j.origin.xyz.0, j.origin.rpy.0);
                let axis = Vec3::from_array(j.axis.xyz.0.map(|v| v as f32)).normalize_or_zero();
                let movable = matches!(j.joint_type, J::Revolute | J::Continuous | J::Prismatic);
                let joint = if let Some(&value) = cfg.lock_joints.get(&j.name) {
                    let v = value as f32;
                    let motion = if matches!(j.joint_type, J::Prismatic) {
                        Transform { rot: Mat3::IDENTITY, trans: axis * v }
                    } else {
                        Transform { rot: Mat3::from_axis_angle(axis, v), trans: Vec3::ZERO }
                    };
                    origin = origin.mul(&motion);
                    JointKind::Fixed
                } else if movable {
                    if j.mimic.is_some() {
                        bail!("mimic joint '{}' is not supported; lock it in the config", j.name);
                    }
                    let dof = dof_names.len();
                    if dof >= MAX_DOF {
                        bail!("more than {MAX_DOF} actuated joints");
                    }
                    let (lo, hi) = if matches!(j.joint_type, J::Continuous) {
                        (-std::f32::consts::PI, std::f32::consts::PI)
                    } else {
                        (j.limit.lower as f32, j.limit.upper as f32)
                    };
                    dof_names.push(j.name.clone());
                    lower.push(lo);
                    upper.push(hi);
                    max_velocity.push(j.limit.velocity as f32);
                    if matches!(j.joint_type, J::Prismatic) {
                        JointKind::Prismatic { dof, axis }
                    } else {
                        JointKind::Revolute { dof, axis }
                    }
                } else if matches!(j.joint_type, J::Fixed) {
                    JointKind::Fixed
                } else {
                    bail!("joint '{}' has unsupported type {:?}", j.name, j.joint_type);
                };
                let dof_mask = match joint {
                    JointKind::Fixed => parent_mask,
                    JointKind::Revolute { dof, .. } | JointKind::Prismatic { dof, .. } => parent_mask | 1 << dof,
                };
                links.push(Link { name: j.child.link.clone(), parent: Some(i), origin, joint, dof_mask });
            }
            i += 1;
        }
        if links.len() > MAX_LINKS {
            bail!("{} links exceeds MAX_LINKS={MAX_LINKS}", links.len());
        }

        let index: HashMap<&str, usize> = links.iter().enumerate().map(|(i, l)| (l.name.as_str(), i)).collect();
        let ee_link = *index.get(cfg.ee_link.as_str()).with_context(|| format!("unknown ee_link '{}'", cfg.ee_link))?;
        if let Some(name) = cfg.spheres.keys().find(|n| !index.contains_key(n.as_str())) {
            bail!("spheres given for link '{name}', which is not reachable from '{}'", cfg.base_link);
        }
        let mut spheres = vec![];
        for (li, link) in links.iter().enumerate() {
            let buffer = cfg.self_collision_buffer.get(&link.name).copied().unwrap_or(0.0);
            for s in cfg.spheres.get(&link.name).into_iter().flatten() {
                spheres.push(CollisionSphere {
                    link: li,
                    center: Vec3::new(s[0], s[1], s[2]),
                    radius: s[3],
                    self_buffer: buffer,
                });
            }
        }
        if spheres.len() > MAX_SPHERES {
            bail!("{} spheres exceeds MAX_SPHERES={MAX_SPHERES}", spheres.len());
        }

        // Self-collision: every sphere pair on different links, minus ignored link pairs.
        // Ignore entries naming links absent from the URDF (e.g. attached objects) are skipped.
        let mut ignore = HashSet::new();
        for (a, others) in &cfg.self_collision_ignore {
            for b in others {
                if let (Some(&ia), Some(&ib)) = (index.get(a.as_str()), index.get(b.as_str())) {
                    ignore.insert((ia.min(ib), ia.max(ib)));
                }
            }
        }
        let mut self_pairs = vec![];
        for a in 0..spheres.len() {
            for b in a + 1..spheres.len() {
                let (la, lb) = (spheres[a].link, spheres[b].link);
                if la != lb && !ignore.contains(&(la.min(lb), la.max(lb))) {
                    self_pairs.push([a as u32, b as u32]);
                }
            }
        }

        if cfg.default_q.len() != dof_names.len() {
            bail!("default_q has {} values but the robot has {} actuated joints", cfg.default_q.len(), dof_names.len());
        }
        Ok(Self {
            name: urdf.name.clone(),
            links,
            dof_names,
            lower,
            upper,
            max_velocity,
            spheres,
            self_pairs,
            ee_link,
            default_q: cfg.default_q.clone(),
        })
    }

    /// The URDF robot name.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dof(&self) -> usize {
        self.dof_names.len()
    }

    pub fn joint_names(&self) -> &[String] {
        &self.dof_names
    }

    pub fn lower(&self) -> &[f32] {
        &self.lower
    }

    pub fn upper(&self) -> &[f32] {
        &self.upper
    }

    pub fn max_velocity(&self) -> &[f32] {
        &self.max_velocity
    }

    pub fn default_q(&self) -> &[f32] {
        &self.default_q
    }

    /// World pose of the IK target frame at configuration `q`.
    pub fn ee_pose(&self, q: &[f32]) -> Pose {
        self.pose_of(&self.fk(q), self.ee_link)
    }

    /// World pose of the named link at configuration `q`.
    pub fn link_pose(&self, q: &[f32], link: &str) -> Option<Pose> {
        let i = self.links.iter().position(|l| l.name == link)?;
        Some(self.pose_of(&self.fk(q), i))
    }

    fn pose_of(&self, fk: &Fk, link: usize) -> Pose {
        Pose { position: fk.pos[link], rotation: Quat::from_mat3(&fk.rot[link]) }
    }

    pub(crate) fn fk(&self, q: &[f32]) -> Fk {
        let mut fk = Fk {
            rot: [Mat3::IDENTITY; MAX_LINKS],
            pos: [Vec3::ZERO; MAX_LINKS],
            axis: [Vec3::ZERO; MAX_DOF],
            anchor: [Vec3::ZERO; MAX_DOF],
            prismatic: 0,
        };
        for (i, link) in self.links.iter().enumerate() {
            let (prot, ppos) = link.parent.map_or((Mat3::IDENTITY, Vec3::ZERO), |p| (fk.rot[p], fk.pos[p]));
            let jrot = prot * link.origin.rot;
            let jpos = prot * link.origin.trans + ppos;
            match link.joint {
                JointKind::Fixed => {
                    fk.rot[i] = jrot;
                    fk.pos[i] = jpos;
                }
                JointKind::Revolute { dof, axis } => {
                    fk.rot[i] = jrot * Mat3::from_axis_angle(axis, q[dof]);
                    fk.pos[i] = jpos;
                    fk.axis[dof] = jrot * axis;
                    fk.anchor[dof] = jpos;
                }
                JointKind::Prismatic { dof, axis } => {
                    let a = jrot * axis;
                    fk.rot[i] = jrot;
                    fk.pos[i] = jpos + a * q[dof];
                    fk.axis[dof] = a;
                    fk.anchor[dof] = jpos;
                    fk.prismatic |= 1 << dof;
                }
            }
        }
        fk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panda() -> Robot {
        Robot::from_config_file(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/franka/panda.json")).unwrap()
    }

    #[test]
    fn self_collision_pairs_skip_same_and_ignored_links() {
        let robot = panda();
        assert_eq!(robot.spheres.len(), 61);
        let link = |name: &str| robot.links.iter().position(|l| l.name == name).unwrap();
        let pairs_between = |a: usize, b: usize| {
            robot
                .self_pairs
                .iter()
                .filter(|&&[x, y]| {
                    let (lx, ly) = (robot.spheres[x as usize].link, robot.spheres[y as usize].link);
                    (lx, ly) == (a, b) || (lx, ly) == (b, a)
                })
                .count()
        };
        assert_eq!(pairs_between(link("panda_link1"), link("panda_link2")), 0);
        assert_eq!(pairs_between(link("panda_link0"), link("panda_link5")), 2 * 13);
        assert!(
            robot.self_pairs.iter().all(|&[a, b]| robot.spheres[a as usize].link != robot.spheres[b as usize].link)
        );
    }
}
