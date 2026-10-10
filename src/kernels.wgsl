// Batched kinematics, collision cost/gradient, IK and trajectory optimization.
// Mirrors src/cpu.rs function for function. gpu.rs prepends the constants (MAX_DOF, MAX_LINKS,
// MAX_JOINTS, MAX_SPHERES, JAC_LEN, obstacle kinds) and the shared structs (Params, Link, Sphere,
// Obstacle, Iter), generated from their Rust definitions so the two sides cannot drift apart.

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
// trajopt: [collision gradient per sample | gradient | Adam m | Adam v] (see sample_grad_len).
@group(0) @binding(11) var<storage, read_write> aux: array<f32>;
@group(0) @binding(12) var<storage, read_write> outbuf: array<f32>;
@group(1) @binding(0) var<uniform> IT: Iter;

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

// The trajopt part of aux: [collision gradient per sample | gradient | Adam m | Adam v].
fn sample_grad_len() -> u32 {
    return P.n_items * (P.points - 3u) * P.samples * P.n_dof;
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
    collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin);
    for (var j = 0u; j < n; j++) {
        aux[idx * n + j] = grad[j];
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
            let g = ((item * spans + t - i) * P.samples + s) * n;
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
        aux[sample_grad_len() + base + j] = grad[j] + g;
    }
}

// One invocation per free control point: Adam step from the gradient, clamped to joint limits.
@compute @workgroup_size(64)
fn traj_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let tn = P.points;
    let free = tn - 6u;
    if (idx >= P.n_items * free) {
        return;
    }
    let item = idx / free;
    let t = idx % free + 3u;
    let n = P.n_dof;
    let base = (item * tn + t) * n;
    let total = P.n_items * tn * n;
    let g0 = sample_grad_len();
    for (var j = 0u; j < n; j++) {
        let i = base + j;
        let g = aux[g0 + i];
        let m = P.beta1 * aux[g0 + total + i] + (1.0 - P.beta1) * g;
        let v = P.beta2 * aux[g0 + 2u * total + i] + (1.0 - P.beta2) * g * g;
        aux[g0 + total + i] = m;
        aux[g0 + 2u * total + i] = v;
        let step = IT.lr * (m / IT.bc1) / (sqrt(v / IT.bc2) + IT.eps);
        let lim = limits[j];
        qbuf[i] = clamp(qbuf[i] - step, lim.x, lim.y);
    }
}
