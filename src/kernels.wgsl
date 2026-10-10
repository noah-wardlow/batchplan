// Batched kinematics, collision cost/gradient, IK and trajectory optimization.
// Mirrors src/cpu.rs function for function. gpu.rs prepends the constants (MAX_DOF, MAX_LINKS,
// MAX_JOINTS, MAX_SPHERES, JAC_LEN, obstacle kinds, MAX_HISTORY, LINE_STEPS and line_search) and
// the shared structs (Params, Link, Sphere, Obstacle), generated from their Rust definitions so the
// two sides cannot drift apart.

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
// Distance grid values, two half floats per word (see Obstacle.grid).
@group(0) @binding(9) var<storage, read> grid_data: array<u32>;
@group(0) @binding(10) var<storage, read_write> qbuf: array<f32>;
// trajopt: the L-BFGS state of each path (see lbfgs_stride).
@group(0) @binding(11) var<storage, read_write> aux: array<f32>;
@group(0) @binding(12) var<storage, read_write> outbuf: array<f32>;

const FAR: f32 = 1e30;

var<private> lrot: array<mat3x3<f32>, MAX_LINKS>;
var<private> lpos: array<vec3<f32>, MAX_LINKS>;
// World axis and anchor of each moving joint (numbered by `Link.joint`); bit j of `prismatic` is
// set when joint j slides. Kept to MAX_JOINTS entries so they stay in registers.
var<private> jaxis: array<vec3<f32>, MAX_JOINTS>;
var<private> janchor: array<vec3<f32>, MAX_JOINTS>;
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
            lrot[i] = jrot * rodrigues(l.axis.xyz, l.axis.w * q[l.dof] + l.trans.w);
            lpos[i] = jpos;
            jaxis[l.joint] = jrot * l.axis.xyz;
            janchor[l.joint] = jpos;
        } else if (l.kind == 2u) {
            let a = jrot * l.axis.xyz;
            lrot[i] = jrot;
            lpos[i] = jpos + a * (l.axis.w * q[l.dof] + l.trans.w);
            jaxis[l.joint] = a;
            janchor[l.joint] = jpos;
            prismatic |= 1u << l.joint;
        } else {
            lrot[i] = jrot;
            lpos[i] = jpos;
        }
    }
}

// Derivative of a point rigidly attached downstream of moving joint j, per unit of joint motion.
fn dpoint(j: u32, p: vec3<f32>) -> vec3<f32> {
    if (((prismatic >> j) & 1u) == 1u) {
        return jaxis[j];
    }
    return cross(jaxis[j], p - janchor[j]);
}

fn grid_value(o: Obstacle, n: u32) -> f32 {
    let pair = unpack2x16float(grid_data[o.grid + (n >> 1u)]);
    return select(pair.x, pair.y, (n & 1u) == 1u);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Returns (gradient, signed distance) at p in the grid's frame. Mirrors grid_distance in sdf.rs.
fn grid_distance(o: Obstacle, p: vec3<f32>) -> vec4<f32> {
    let voxel = o.half.w;
    let top = vec3<f32>(f32(o.nx), f32(o.ny), f32(o.nz)) - 1.0;
    let x = (p - o.half.xyz) / voxel;
    let c = clamp(x, vec3<f32>(0.0), top);
    let i = min(floor(c), top - 1.0);
    let f = c - i;
    let sy = o.nx;
    let sz = o.nx * o.ny;
    let n = u32(i.x) + sy * u32(i.y) + sz * u32(i.z);
    let v000 = grid_value(o, n);
    let v100 = grid_value(o, n + 1u);
    let v010 = grid_value(o, n + sy);
    let v110 = grid_value(o, n + sy + 1u);
    let v001 = grid_value(o, n + sz);
    let v101 = grid_value(o, n + sz + 1u);
    let v011 = grid_value(o, n + sz + sy);
    let v111 = grid_value(o, n + sz + sy + 1u);
    let x00 = lerp(v000, v100, f.x);
    let x10 = lerp(v010, v110, f.x);
    let x01 = lerp(v001, v101, f.x);
    let x11 = lerp(v011, v111, f.x);
    let y0 = lerp(x00, x10, f.y);
    let y1 = lerp(x01, x11, f.y);
    let dx = lerp(lerp(v100 - v000, v110 - v010, f.y), lerp(v101 - v001, v111 - v011, f.y), f.z);
    let inside = vec3<f32>(dx, lerp(x10 - x00, x11 - x01, f.z), y1 - y0) / voxel;
    let outside = (x - c) * voxel;
    let away = length(outside);
    let g = select(inside, outside / max(away, 1e-30), x != c);
    return vec4<f32>(g, lerp(y0, y1, f.z) + away);
}

// Returns (gradient direction, signed distance). Mirrors the distance functions in world.rs.
fn obstacle_distance(o: Obstacle, p: vec3<f32>) -> vec4<f32> {
    let v = p - o.center.xyz;
    let kind = u32(o.center.w);
    if (kind == SPHERE) {
        let len = length(v);
        var g = vec3<f32>(0.0, 0.0, 1.0);
        if (len > 1e-9) {
            g = v / len;
        }
        return vec4<f32>(g, len - o.half.x);
    }
    let lp = vec3<f32>(dot(o.r0.xyz, v), dot(o.r1.xyz, v), dot(o.r2.xyz, v));
    var gl: vec3<f32>;
    var d: f32;
    if (kind == CUBOID) {
        let sgn = select(vec3<f32>(-1.0), vec3<f32>(1.0), lp >= vec3<f32>(0.0));
        let qd = abs(lp) - o.half.xyz;
        let outside = max(qd, vec3<f32>(0.0));
        let olen = length(outside);
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
    } else if (kind == CYLINDER) {
        let rho = sqrt(lp.x * lp.x + lp.y * lp.y);
        var radial = vec3<f32>(1.0, 0.0, 0.0);
        if (rho > 1e-9) {
            radial = vec3<f32>(lp.x / rho, lp.y / rho, 0.0);
        }
        let axial = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, lp.z >= 0.0));
        let dr = rho - o.half.x;
        let dz = abs(lp.z) - o.half.y;
        if (dr > 0.0 && dz > 0.0) {
            d = sqrt(dr * dr + dz * dz);
            gl = (radial * dr + axial * dz) / d;
        } else if (dr >= dz) {
            d = dr;
            gl = radial;
        } else {
            d = dz;
            gl = axial;
        }
    } else if (kind == SDF) {
        let gd = grid_distance(o, lp);
        d = gd.w;
        gl = gd.xyz;
    } else {
        let w = lp - vec3<f32>(0.0, 0.0, clamp(lp.z, -o.half.y, o.half.y));
        let len = length(w);
        d = len - o.half.x;
        gl = vec3<f32>(1.0, 0.0, 0.0);
        if (len > 1e-9) {
            gl = w / len;
        }
    }
    return vec4<f32>(o.r0.xyz * gl.x + o.r1.xyz * gl.y + o.r2.xyz * gl.z, d);
}

// Collision cost of the configuration last passed to fk(); with `gradient`, adds d(cost)/dq into
// `grad`. Returns (cost, world clearance, self clearance). Callers pass `gradient` as a constant,
// so the compiler drops the per-sphere gradient work from cost-only kernels.
fn collision(world: u32, w_world: f32, w_self: f32, margin: f32, self_margin: f32, gradient: bool) -> vec3<f32> {
    let ns = P.n_spheres;
    for (var s = 0u; s < ns; s++) {
        let sp = spheres[s];
        sc[s] = lrot[sp.link] * sp.c.xyz + lpos[sp.link];
        if (gradient) {
            gc[s] = vec3<f32>(0.0);
        }
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
                if (gradient) {
                    gc[s] -= 2.0 * w_world * pen * dg.xyz;
                }
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
        if (pen > 0.0 && gradient) {
            var u = vec3<f32>(1.0, 0.0, 0.0);
            if (dist > 1e-9) {
                u = diff / dist;
            }
            let g = 2.0 * w_self * pen * u;
            gc[pr.x] -= g;
            gc[pr.y] += g;
        }
        if (pen > 0.0) {
            cost += w_self * pen * pen;
        }
    }
    if (!gradient) {
        return vec3<f32>(cost, wmin, smin);
    }
    for (var s = 0u; s < ns; s++) {
        let g = gc[s];
        if (all(g == vec3<f32>(0.0))) {
            continue;
        }
        // Every moving joint at or above the sphere's link, mimic joints adding into their leader.
        var chain = links[spheres[s].link].chain;
        while (chain != 0u) {
            let i = firstTrailingBit(chain);
            chain &= chain - 1u;
            grad[links[i].dof] += links[i].axis.w * dot(g, dpoint(links[i].joint, sc[s]));
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
    for (var k = 0u; k < JAC_LEN; k++) {
        jac[k] = 0.0;
    }
    var chain = links[ee].chain;
    while (chain != 0u) {
        let i = firstTrailingBit(chain);
        chain &= chain - 1u;
        {
            let j = links[i].dof;
            let k = links[i].joint;
            let m = links[i].axis.w;
            let jp = m * dpoint(k, lpos[ee]);
            var jo = vec3<f32>(0.0);
            if (((prismatic >> k) & 1u) == 0u) {
                jo = m * jaxis[k] * P.rot_weight;
            }
            jac[0u * MAX_DOF + j] += jp.x;
            jac[1u * MAX_DOF + j] += jp.y;
            jac[2u * MAX_DOF + j] += jp.z;
            jac[3u * MAX_DOF + j] += jo.x;
            jac[4u * MAX_DOF + j] += jo.y;
            jac[5u * MAX_DOF + j] += jo.z;
        }
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
        collision(world, P.w_world, P.w_self, P.margin, P.self_margin, true);
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
    let c = collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, true);
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

// Uniform cubic B-spline weights of a span's four control points at u in [0, 1]. Mirrors spline.rs.
fn basis(u: f32) -> vec4<f32> {
    let v = 1.0 - u;
    let u2 = u * u;
    let u3 = u2 * u;
    return vec4<f32>(v * v * v, 3.0 * u3 - 6.0 * u2 + 4.0, -3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0, u3) / 6.0;
}

// Collision sample s of a span sits at u = (s + 0.5) / P.samples.
fn sample_u(s: u32) -> f32 {
    return (f32(s) + 0.5) / f32(P.samples);
}

// Collision samples along one path.
fn path_samples() -> u32 {
    return (P.points - 3u) * P.samples;
}

// Each path's L-BFGS state in aux, in this order: the collision gradient per sample, the
// collision cost per line-search step and sample, then per control point the gradient, the
// previous gradient, the direction, the pending step, the step history and the gradient-change
// history, then five scalars: cost, history count, newest slot, started, pending. Mirrors
// lbfgs_stride in gpu.rs and the Lbfgs struct in cpu.rs.
fn lbfgs_stride() -> u32 {
    let tn = P.points * P.n_dof;
    return path_samples() * P.n_dof + LINE_STEPS * path_samples() + (4u + 2u * P.history) * tn + 5u;
}

fn costs_at(item: u32) -> u32 {
    return item * lbfgs_stride() + path_samples() * P.n_dof;
}

fn grad_at(item: u32) -> u32 {
    return costs_at(item) + LINE_STEPS * path_samples();
}

fn prev_grad_at(item: u32) -> u32 {
    return grad_at(item) + P.points * P.n_dof;
}

fn dir_at(item: u32) -> u32 {
    return prev_grad_at(item) + P.points * P.n_dof;
}

fn pending_step_at(item: u32) -> u32 {
    return dir_at(item) + P.points * P.n_dof;
}

fn steps_at(item: u32, slot: u32) -> u32 {
    return pending_step_at(item) + (1u + slot) * P.points * P.n_dof;
}

fn changes_at(item: u32, slot: u32) -> u32 {
    return steps_at(item, P.history + slot);
}

fn scalars_at(item: u32) -> u32 {
    return steps_at(item, 2u * P.history);
}

// Control point t, joint j of the candidate path x + alpha d: free points move and are clamped
// to the joint range, the three pinned at each end stay.
fn candidate(item: u32, t: u32, j: u32, alpha: f32) -> f32 {
    let i = t * P.n_dof + j;
    let x = qbuf[item * P.points * P.n_dof + i];
    if (t < 3u || t >= P.points - 3u) {
        return x;
    }
    let lim = limits[j];
    return clamp(x + alpha * aux[dir_at(item) + i], lim.x, lim.y);
}

// One invocation per line-search step and collision sample: the collision cost at that point of
// the candidate path.
@compute @workgroup_size(64)
fn traj_costs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let per_item = path_samples();
    if (idx >= P.n_items * LINE_STEPS * per_item) {
        return;
    }
    let item = idx / (LINE_STEPS * per_item);
    let c = idx / per_item % LINE_STEPS;
    let sample = idx % per_item;
    let span = sample / P.samples;
    let w = basis(sample_u(sample % P.samples));
    let alpha = line_search(c);
    for (var j = 0u; j < P.n_dof; j++) {
        q[j] = w.x * candidate(item, span, j, alpha) + w.y * candidate(item, span + 1u, j, alpha)
            + w.z * candidate(item, span + 2u, j, alpha) + w.w * candidate(item, span + 3u, j, alpha);
    }
    fk();
    let c3 = collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, false);
    aux[costs_at(item) + c * per_item + sample] = c3.x;
}

// One invocation per path: the trajectory cost of each line-search step; the cheapest moves the
// path if it lowers the cost, otherwise the history resets.
@compute @workgroup_size(64)
fn traj_search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let item = item_index(gid, nwg);
    if (item >= P.n_items) {
        return;
    }
    let n = P.n_dof;
    let tn = P.points;
    let per_item = path_samples();
    let sc = scalars_at(item);
    let started = aux[sc + 3u] > 0.5;
    var cost = aux[sc];
    var best = -1;
    for (var c = 0u; c < LINE_STEPS; c++) {
        let alpha = line_search(c);
        var total = 0.0;
        for (var s = 0u; s < per_item; s++) {
            total += aux[costs_at(item) + c * per_item + s];
        }
        for (var j = 0u; j < n; j++) {
            // Each candidate point once, sliding along the path.
            var before = 0.0;
            var previous = candidate(item, 0u, j, alpha);
            for (var t = 1u; t < tn; t++) {
                let current = candidate(item, t, j, alpha);
                let v = current - previous;
                total += P.w_vel * v * v;
                if (t >= 2u) {
                    let a = current - 2.0 * previous + before;
                    total += P.w_acc * a * a;
                }
                before = previous;
                previous = current;
            }
        }
        if ((best < 0 && !started) || total < cost) {
            cost = total;
            best = i32(c);
        }
    }
    if (best < 0) {
        aux[sc + 4u] = 0.0;
        aux[sc + 1u] = 0.0;
        return;
    }
    let alpha = line_search(u32(best));
    for (var t = 3u; t < tn - 3u; t++) {
        for (var j = 0u; j < n; j++) {
            let i = t * n + j;
            let moved = candidate(item, t, j, alpha);
            let x = item * tn * n + i;
            aux[pending_step_at(item) + i] = moved - qbuf[x];
            qbuf[x] = moved;
        }
    }
    aux[sc] = cost;
    aux[sc + 3u] = 1.0;
    aux[sc + 4u] = 1.0;
}

fn lbfgs_dot(a: u32, b: u32) -> f32 {
    var sum = 0.0;
    for (var i = 3u * P.n_dof; i < (P.points - 3u) * P.n_dof; i++) {
        sum += aux[a + i] * aux[b + i];
    }
    return sum;
}

// One invocation per path: records the last step and the gradient change it caused, then the
// L-BFGS two-loop recursion for the next direction; steepest descent scaled to initial_step
// without history or when the recursion fails to descend.
@compute @workgroup_size(64)
fn lbfgs_direction(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let item = item_index(gid, nwg);
    if (item >= P.n_items) {
        return;
    }
    let first = 3u * P.n_dof;
    let end = (P.points - 3u) * P.n_dof;
    let m = P.history;
    let sc = scalars_at(item);
    let g = grad_at(item);
    let gp = prev_grad_at(item);
    let d = dir_at(item);
    var count = u32(aux[sc + 1u]);
    var newest = u32(aux[sc + 2u]);
    if (aux[sc + 4u] > 0.5) {
        let s = pending_step_at(item);
        var sy = 0.0;
        for (var i = first; i < end; i++) {
            sy += aux[s + i] * (aux[g + i] - aux[gp + i]);
        }
        if (sy > 1e-10) {
            var slot = 0u;
            if (count > 0u) {
                slot = (newest + 1u) % m;
            }
            for (var i = first; i < end; i++) {
                aux[steps_at(item, slot) + i] = aux[s + i];
                aux[changes_at(item, slot) + i] = aux[g + i] - aux[gp + i];
            }
            newest = slot;
            count = min(count + 1u, m);
        }
        aux[sc + 4u] = 0.0;
    }
    var largest = 0.0;
    for (var i = first; i < end; i++) {
        aux[gp + i] = aux[g + i];
        aux[d + i] = -aux[g + i];
        largest = max(largest, abs(aux[g + i]));
    }
    var alpha: array<f32, MAX_HISTORY>;
    for (var a = 0u; a < count; a++) {
        let slot = (newest + m - a) % m;
        let s = steps_at(item, slot);
        let y = changes_at(item, slot);
        alpha[a] = lbfgs_dot(s, d) / lbfgs_dot(y, s);
        for (var i = first; i < end; i++) {
            aux[d + i] -= alpha[a] * aux[y + i];
        }
    }
    var gamma = 0.0;
    if (count > 0u) {
        let s = steps_at(item, newest);
        let y = changes_at(item, newest);
        gamma = lbfgs_dot(s, y) / lbfgs_dot(y, y);
    } else if (largest > 0.0) {
        gamma = P.initial_step / largest;
    }
    for (var i = first; i < end; i++) {
        aux[d + i] *= gamma;
    }
    for (var k = 0u; k < count; k++) {
        let a = count - 1u - k;
        let slot = (newest + m - a) % m;
        let s = steps_at(item, slot);
        let y = changes_at(item, slot);
        let beta = lbfgs_dot(y, d) / lbfgs_dot(y, s);
        for (var i = first; i < end; i++) {
            aux[d + i] += (alpha[a] - beta) * aux[s + i];
        }
    }
    if (largest > 0.0 && lbfgs_dot(d, g) >= 0.0) {
        count = 0u;
        for (var i = first; i < end; i++) {
            aux[d + i] = -aux[g + i] * (P.initial_step / largest);
        }
    }
    aux[sc + 1u] = f32(count);
    aux[sc + 2u] = f32(newest);
}

// One invocation per collision sample: writes d(collision cost)/dq at that point of the curve.
@compute @workgroup_size(64)
fn traj_samples(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let per_item = (P.points - 3u) * P.samples;
    if (idx >= P.n_items * per_item) {
        return;
    }
    let item = idx / per_item;
    let span = (idx % per_item) / P.samples;
    let w = basis(sample_u(idx % P.samples));
    let n = P.n_dof;
    let base = (item * P.points + span) * n;
    for (var j = 0u; j < n; j++) {
        q[j] = w.x * qbuf[base + j] + w.y * qbuf[base + n + j] + w.z * qbuf[base + 2u * n + j] + w.w * qbuf[base + 3u * n + j];
        grad[j] = 0.0;
    }
    fk();
    collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, true);
    let sample = idx % per_item;
    for (var j = 0u; j < n; j++) {
        aux[item * lbfgs_stride() + sample * n + j] = grad[j];
    }
}

// One invocation per free control point: the collision gradients of the samples it shapes, each
// weighted by its basis value there, plus smoothness on the control points.
@compute @workgroup_size(64)
fn traj_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let tn = P.points;
    let free = tn - 6u;
    if (idx >= P.n_items * free) {
        return;
    }
    let item = idx / free;
    let t = idx % free + 3u;
    let n = P.n_dof;
    let spans = tn - 3u;
    let base = (item * tn + t) * n;
    for (var j = 0u; j < n; j++) {
        grad[j] = 0.0;
    }
    // Control point t is point i of span t - i.
    for (var i = 0u; i < 4u; i++) {
        if (t < i || t - i >= spans) {
            continue;
        }
        for (var s = 0u; s < P.samples; s++) {
            let w = basis(sample_u(s))[i];
            let g = item * lbfgs_stride() + ((t - i) * P.samples + s) * n;
            for (var j = 0u; j < n; j++) {
                grad[j] += w * aux[g + j];
            }
        }
    }
    for (var j = 0u; j < n; j++) {
        let qm = qbuf[base - n + j];
        let q0 = qbuf[base + j];
        let qp = qbuf[base + n + j];
        var g = 2.0 * P.w_vel * (2.0 * q0 - qm - qp);
        g += 2.0 * P.w_acc * -2.0 * (qp - 2.0 * q0 + qm);
        if (t >= 2u) {
            g += 2.0 * P.w_acc * (q0 - 2.0 * qm + qbuf[base - 2u * n + j]);
        }
        if (t + 2u < tn) {
            g += 2.0 * P.w_acc * (qbuf[base + 2u * n + j] - 2.0 * qp + q0);
        }
        aux[grad_at(item) + t * n + j] = grad[j] + g;
    }
}
