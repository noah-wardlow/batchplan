// Batched kinematics, collision cost/gradient, IK and trajectory optimization.
// Mirrors src/cpu.rs function for function. MAX_DOF, MAX_LINKS and MAX_SPHERES are
// prepended by gpu.rs from the Rust constants.

struct Params {
    n_dof: u32, n_links: u32, n_spheres: u32, n_pairs: u32,
    ee_link: u32, n_items: u32, waypoints: u32, iterations: u32,
    w_world: f32, w_self: f32, margin: f32, self_margin: f32,
    w_acc: f32, w_vel: f32, beta1: f32, beta2: f32,
    damping: f32, rot_weight: f32, max_step: f32, collision_step: f32,
}

struct Link {
    c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>,  // origin rotation columns
    trans: vec4<f32>,
    axis: vec4<f32>,
    parent: i32, kind: u32, dof: u32, mask: u32,  // kind: 0 fixed, 1 revolute, 2 prismatic
}

struct Sphere { c: vec4<f32>, link: u32, self_buf: f32, pad0: u32, pad1: u32 }

// center.w: 0 = cuboid, 1 = sphere (radius in half.x). r0..r2: world-from-box rotation columns.
struct Obstacle { center: vec4<f32>, half: vec4<f32>, r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> }

struct Iter { lr: f32, bc1: f32, bc2: f32, pad: f32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> links: array<Link>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> limits: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(6) var<storage, read> world_ranges: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> item_world: array<u32>;
// IK targets: 4 vec4 per item (position, rotation columns).
@group(0) @binding(8) var<storage, read> targets: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> qbuf: array<f32>;
// trajopt: [grad | adam m | adam v], each the size of qbuf.
@group(0) @binding(10) var<storage, read_write> aux: array<f32>;
@group(0) @binding(11) var<storage, read_write> outbuf: array<f32>;
@group(1) @binding(0) var<uniform> IT: Iter;

const FAR: f32 = 1e30;

var<private> lrot: array<mat3x3<f32>, MAX_LINKS>;
var<private> lpos: array<vec3<f32>, MAX_LINKS>;
var<private> jaxis: array<vec3<f32>, MAX_DOF>;
var<private> janchor: array<vec3<f32>, MAX_DOF>;
var<private> prismatic: u32;
var<private> q: array<f32, MAX_DOF>;
var<private> grad: array<f32, MAX_DOF>;
var<private> sc: array<vec3<f32>, MAX_SPHERES>;
var<private> gc: array<vec3<f32>, MAX_SPHERES>;
var<private> jac: array<f32, JAC_LEN>;  // 6 x MAX_DOF, row-major
var<private> chol: array<f32, 36>;

fn item_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn rodrigues(a: vec3<f32>, angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let t = 1.0 - c;
    return mat3x3<f32>(
        vec3<f32>(t * a.x * a.x + c, t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y),
        vec3<f32>(t * a.x * a.y - s * a.z, t * a.y * a.y + c, t * a.y * a.z + s * a.x),
        vec3<f32>(t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c));
}

fn fk() {
    prismatic = 0u;
    for (var i = 0u; i < P.n_links; i++) {
        let l = links[i];
        var prot = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
        var ppos = vec3<f32>(0.0);
        if (l.parent >= 0) {
            prot = lrot[l.parent];
            ppos = lpos[l.parent];
        }
        let jrot = prot * mat3x3<f32>(l.c0.xyz, l.c1.xyz, l.c2.xyz);
        let jpos = prot * l.trans.xyz + ppos;
        if (l.kind == 1u) {
            lrot[i] = jrot * rodrigues(l.axis.xyz, q[l.dof]);
            lpos[i] = jpos;
            jaxis[l.dof] = jrot * l.axis.xyz;
            janchor[l.dof] = jpos;
        } else if (l.kind == 2u) {
            let a = jrot * l.axis.xyz;
            lrot[i] = jrot;
            lpos[i] = jpos + a * q[l.dof];
            jaxis[l.dof] = a;
            janchor[l.dof] = jpos;
            prismatic |= 1u << l.dof;
        } else {
            lrot[i] = jrot;
            lpos[i] = jpos;
        }
    }
}

fn dpoint(j: u32, p: vec3<f32>) -> vec3<f32> {
    if (((prismatic >> j) & 1u) == 1u) {
        return jaxis[j];
    }
    return cross(jaxis[j], p - janchor[j]);
}

// Returns (gradient direction, signed distance).
fn obstacle_distance(o: Obstacle, p: vec3<f32>) -> vec4<f32> {
    let v = p - o.center.xyz;
    if (o.center.w > 0.5) {
        let len = length(v);
        var g = vec3<f32>(0.0, 0.0, 1.0);
        if (len > 1e-9) {
            g = v / len;
        }
        return vec4<f32>(g, len - o.half.x);
    }
    let lp = vec3<f32>(dot(o.r0.xyz, v), dot(o.r1.xyz, v), dot(o.r2.xyz, v));
    let sgn = select(vec3<f32>(-1.0), vec3<f32>(1.0), lp >= vec3<f32>(0.0));
    let qd = abs(lp) - o.half.xyz;
    let outside = max(qd, vec3<f32>(0.0));
    let olen = length(outside);
    var gl: vec3<f32>;
    var d: f32;
    if (olen > 0.0) {
        d = olen;
        gl = sgn * outside / olen;
    } else if (qd.x >= qd.y && qd.x >= qd.z) {
        d = qd.x;
        gl = vec3<f32>(sgn.x, 0.0, 0.0);
    } else if (qd.y >= qd.z) {
        d = qd.y;
        gl = vec3<f32>(0.0, sgn.y, 0.0);
    } else {
        d = qd.z;
        gl = vec3<f32>(0.0, 0.0, sgn.z);
    }
    return vec4<f32>(o.r0.xyz * gl.x + o.r1.xyz * gl.y + o.r2.xyz * gl.z, d);
}

// Collision cost of the configuration last passed to fk(); adds d(cost)/dq into `grad`.
// Returns (cost, world clearance, self clearance).
fn collision(world: u32, w_world: f32, w_self: f32, margin: f32, self_margin: f32) -> vec3<f32> {
    let ns = P.n_spheres;
    for (var s = 0u; s < ns; s++) {
        let sp = spheres[s];
        sc[s] = lrot[sp.link] * sp.c.xyz + lpos[sp.link];
        gc[s] = vec3<f32>(0.0);
    }
    var cost = 0.0;
    var wmin = FAR;
    var smin = FAR;
    let range = world_ranges[world];
    for (var s = 0u; s < ns; s++) {
        let r = spheres[s].c.w;
        for (var k = 0u; k < range.y; k++) {
            let dg = obstacle_distance(obstacles[range.x + k], sc[s]);
            let d = dg.w - r;
            wmin = min(wmin, d);
            let pen = margin - d;
            if (pen > 0.0) {
                cost += w_world * pen * pen;
                gc[s] -= 2.0 * w_world * pen * dg.xyz;
            }
        }
    }
    for (var k = 0u; k < P.n_pairs; k++) {
        let pr = pairs[k];
        let a = spheres[pr.x];
        let b = spheres[pr.y];
        let diff = sc[pr.x] - sc[pr.y];
        let dist = length(diff);
        let d = dist - (a.c.w + a.self_buf) - (b.c.w + b.self_buf);
        smin = min(smin, d);
        let pen = self_margin - d;
        if (pen > 0.0) {
            cost += w_self * pen * pen;
            var u = vec3<f32>(1.0, 0.0, 0.0);
            if (dist > 1e-9) {
                u = diff / dist;
            }
            let g = 2.0 * w_self * pen * u;
            gc[pr.x] -= g;
            gc[pr.y] += g;
        }
    }
    for (var s = 0u; s < ns; s++) {
        let g = gc[s];
        if (all(g == vec3<f32>(0.0))) {
            continue;
        }
        let mask = links[spheres[s].link].mask;
        for (var j = 0u; j < P.n_dof; j++) {
            if (((mask >> j) & 1u) == 1u) {
                grad[j] += dot(g, dpoint(j, sc[s]));
            }
        }
    }
    return vec3<f32>(cost, wmin, smin);
}

// Rotation vector and angle of r (column-major: r[col][row]).
fn rot_log(r: mat3x3<f32>) -> vec4<f32> {
    let cos_t = clamp((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5, -1.0, 1.0);
    let v = 0.5 * vec3<f32>(r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]);
    let sv = length(v);
    let theta = atan2(sv, cos_t);
    if (sv > 1e-6) {
        return vec4<f32>(v * (theta / sv), theta);
    }
    if (cos_t > 0.0) {
        return vec4<f32>(v, theta);
    }
    var axis: vec3<f32>;
    if (r[1][1] > r[0][0]) {
        if (r[2][2] > r[1][1]) { axis = r[2] + vec3<f32>(0.0, 0.0, 1.0); } else { axis = r[1] + vec3<f32>(0.0, 1.0, 0.0); }
    } else if (r[2][2] > r[0][0]) {
        axis = r[2] + vec3<f32>(0.0, 0.0, 1.0);
    } else {
        axis = r[0] + vec3<f32>(1.0, 0.0, 0.0);
    }
    return vec4<f32>(normalize(axis) * theta, theta);
}

fn chol6() {
    for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = chol[i * 6u + j];
            for (var k = 0u; k < j; k++) {
                s -= chol[i * 6u + k] * chol[j * 6u + k];
            }
            if (i == j) {
                chol[i * 6u + j] = sqrt(max(s, 1e-12));
            } else {
                chol[i * 6u + j] = s / chol[j * 6u + j];
            }
        }
    }
}

fn chol6_solve(b: array<f32, 6>) -> array<f32, 6> {
    var y: array<f32, 6>;
    for (var i = 0u; i < 6u; i++) {
        var s = b[i];
        for (var k = 0u; k < i; k++) {
            s -= chol[i * 6u + k] * y[k];
        }
        y[i] = s / chol[i * 6u + i];
    }
    var x: array<f32, 6>;
    for (var ii = 0; ii < 6; ii++) {
        let i = u32(5 - ii);
        var s = y[i];
        for (var k = i + 1u; k < 6u; k++) {
            s -= chol[k * 6u + i] * x[k];
        }
        x[i] = s / chol[i * 6u + i];
    }
    return x;
}

fn target_pos(item: u32) -> vec3<f32> {
    return targets[item * 4u].xyz;
}

fn target_rot(item: u32) -> mat3x3<f32> {
    return mat3x3<f32>(targets[item * 4u + 1u].xyz, targets[item * 4u + 2u].xyz, targets[item * 4u + 3u].xyz);
}

fn ik_step(world: u32, tp: vec3<f32>, tr: mat3x3<f32>) {
    let n = P.n_dof;
    fk();
    let ee = P.ee_link;
    let ep = tp - lpos[ee];
    let eo = rot_log(tr * transpose(lrot[ee])).xyz * P.rot_weight;
    let e = array<f32, 6>(ep.x, ep.y, ep.z, eo.x, eo.y, eo.z);
    let mask = links[ee].mask;
    for (var j = 0u; j < n; j++) {
        var jp = vec3<f32>(0.0);
        var jo = vec3<f32>(0.0);
        if (((mask >> j) & 1u) == 1u) {
            jp = dpoint(j, lpos[ee]);
            if (((prismatic >> j) & 1u) == 0u) {
                jo = jaxis[j] * P.rot_weight;
            }
        }
        jac[0u * MAX_DOF + j] = jp.x;
        jac[1u * MAX_DOF + j] = jp.y;
        jac[2u * MAX_DOF + j] = jp.z;
        jac[3u * MAX_DOF + j] = jo.x;
        jac[4u * MAX_DOF + j] = jo.y;
        jac[5u * MAX_DOF + j] = jo.z;
    }
    for (var r = 0u; r < 6u; r++) {
        for (var c = 0u; c <= r; c++) {
            var s = 0.0;
            for (var j = 0u; j < n; j++) {
                s += jac[r * MAX_DOF + j] * jac[c * MAX_DOF + j];
            }
            if (r == c) {
                s += P.damping * P.damping;
            }
            chol[r * 6u + c] = s;
            chol[c * 6u + r] = s;
        }
    }
    chol6();
    let y = chol6_solve(e);
    var dq: array<f32, MAX_DOF>;
    for (var j = 0u; j < n; j++) {
        var s = 0.0;
        for (var r = 0u; r < 6u; r++) {
            s += jac[r * MAX_DOF + j] * y[r];
        }
        dq[j] = s;
    }
    if (P.collision_step > 0.0) {
        for (var j = 0u; j < n; j++) {
            grad[j] = 0.0;
        }
        collision(world, P.w_world, P.w_self, P.margin, P.self_margin);
        var jg: array<f32, 6>;
        for (var r = 0u; r < 6u; r++) {
            var s = 0.0;
            for (var j = 0u; j < n; j++) {
                s += jac[r * MAX_DOF + j] * grad[j];
            }
            jg[r] = s;
        }
        let z = chol6_solve(jg);
        for (var j = 0u; j < n; j++) {
            var proj = grad[j];
            for (var r = 0u; r < 6u; r++) {
                proj -= jac[r * MAX_DOF + j] * z[r];
            }
            dq[j] -= P.collision_step * proj;
        }
    }
    var largest = 0.0;
    for (var j = 0u; j < n; j++) {
        largest = max(largest, abs(dq[j]));
    }
    var scale = 1.0;
    if (largest > P.max_step) {
        scale = P.max_step / largest;
    }
    for (var j = 0u; j < n; j++) {
        let lim = limits[j];
        q[j] = clamp(q[j] + dq[j] * scale, lim.x, lim.y);
    }
}

// One invocation per configuration: out = [world clearance, self clearance, cost, grad...].
@compute @workgroup_size(64)
fn evaluate_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let item = item_index(gid, nwg);
    if (item >= P.n_items) {
        return;
    }
    let n = P.n_dof;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[item * n + j];
        grad[j] = 0.0;
    }
    fk();
    let c = collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin);
    let base = item * (3u + n);
    outbuf[base] = c.y;
    outbuf[base + 1u] = c.z;
    outbuf[base + 2u] = c.x;
    for (var j = 0u; j < n; j++) {
        outbuf[base + 3u + j] = grad[j];
    }
}

// One invocation per IK seed; runs P.iterations steps, writes q back and [pos err, rot err].
@compute @workgroup_size(64)
fn ik_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let item = item_index(gid, nwg);
    if (item >= P.n_items) {
        return;
    }
    let n = P.n_dof;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[item * n + j];
    }
    let world = item_world[item];
    let tp = target_pos(item);
    let tr = target_rot(item);
    for (var it = 0u; it < P.iterations; it++) {
        ik_step(world, tp, tr);
    }
    for (var j = 0u; j < n; j++) {
        qbuf[item * n + j] = q[j];
    }
    fk();
    let ee = P.ee_link;
    outbuf[item * 2u] = length(tp - lpos[ee]);
    outbuf[item * 2u + 1u] = rot_log(tr * transpose(lrot[ee])).w;
}

// One invocation per interior waypoint: writes the trajectory-cost gradient into aux[0..N).
@compute @workgroup_size(64)
fn traj_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let tn = P.waypoints;
    let inner = tn - 2u;
    if (idx >= P.n_items * inner) {
        return;
    }
    let item = idx / inner;
    let t = idx % inner + 1u;
    let n = P.n_dof;
    let base = (item * tn + t) * n;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[base + j];
        grad[j] = 0.0;
    }
    fk();
    collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin);
    for (var j = 0u; j < n; j++) {
        let qm = qbuf[base - n + j];
        let q0 = q[j];
        let qp = qbuf[base + n + j];
        var g = 2.0 * P.w_vel * (2.0 * q0 - qm - qp);
        g += 2.0 * P.w_acc * -2.0 * (qp - 2.0 * q0 + qm);
        if (t >= 2u) {
            g += 2.0 * P.w_acc * (q0 - 2.0 * qm + qbuf[base - 2u * n + j]);
        }
        if (t + 2u < tn) {
            g += 2.0 * P.w_acc * (qbuf[base + 2u * n + j] - 2.0 * qp + q0);
        }
        aux[base + j] = grad[j] + g;
    }
}

// One invocation per interior waypoint: Adam step from aux gradient, clamped to joint limits.
@compute @workgroup_size(64)
fn traj_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let tn = P.waypoints;
    let inner = tn - 2u;
    if (idx >= P.n_items * inner) {
        return;
    }
    let item = idx / inner;
    let t = idx % inner + 1u;
    let n = P.n_dof;
    let base = (item * tn + t) * n;
    let total = P.n_items * tn * n;
    for (var j = 0u; j < n; j++) {
        let i = base + j;
        let g = aux[i];
        let m = P.beta1 * aux[total + i] + (1.0 - P.beta1) * g;
        let v = P.beta2 * aux[2u * total + i] + (1.0 - P.beta2) * g * g;
        aux[total + i] = m;
        aux[2u * total + i] = v;
        let step = IT.lr * (m / IT.bc1) / (sqrt(v / IT.bc2) + 1e-8);
        let lim = limits[j];
        qbuf[i] = clamp(qbuf[i] - step, lim.x, lim.y);
    }
}
