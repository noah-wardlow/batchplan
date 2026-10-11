// Camera images: one invocation per pixel of each view casts a ray against the robot's spheres and
// the view's world. Mirrors render.rs (`shade`, `ray_sphere`, `intersect`, `ray_box`,
// `ray_cylinder`, `ray_capsule`, `ray_grid`).
//
// Generated before this file: `RenderParams`, `RenderView`, `Obstacle`, the obstacle kinds,
// `WORKGROUP`, `PALETTE`, `ROBOT`, `BACKGROUND`, `GRID_HIT` and `GRID_STEPS`. grid.wgsl follows.

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(2) var<storage, read> world_ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> grid_data: array<u32>;
@group(0) @binding(4) var<storage, read> views: array<RenderView>;
// The robot's spheres in the world (centre, radius), `R.spheres` per view.
@group(0) @binding(5) var<storage, read> robot_spheres: array<vec4<f32>>;
// Per pixel: red, green and blue bytes, and the depth.
@group(0) @binding(6) var<storage, read_write> colors: array<u32>;
@group(0) @binding(7) var<storage, read_write> depths: array<f32>;

const NO_HIT: f32 = -1.0;

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - center;
    let a = dot(dir, dir);
    let b = dot(dir, oc);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - a * c;
    if disc < 0.0 {
        return NO_HIT;
    }
    let t = (-b - sqrt(disc)) / a;
    return select(NO_HIT, t, t > 0.0);
}

// Directions with a zero component would divide by zero.
fn reciprocal(d: vec3<f32>) -> vec3<f32> {
    return 1.0 / select(d, vec3<f32>(1e-20), abs(d) < vec3<f32>(1e-20));
}

// (normal, t) in the box's frame, or t = NO_HIT.
fn ray_box(p: vec3<f32>, d: vec3<f32>, half: vec3<f32>) -> vec4<f32> {
    let inv = reciprocal(d);
    let t0 = (-half - p) * inv;
    let t1 = (half - p) * inv;
    let near = min(t0, t1);
    let far = max(t0, t1);
    let t = max(max(near.x, near.y), near.z);
    if t > min(min(far.x, far.y), far.z) || t <= 0.0 {
        return vec4<f32>(0.0, 0.0, 1.0, NO_HIT);
    }
    var axis = vec3<f32>(0.0, 0.0, 1.0);
    if near.x >= near.y && near.x >= near.z {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else if near.y >= near.z {
        axis = vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec4<f32>(axis * -sign(dot(d, axis)), t);
}

// The side of a cylinder or capsule of `radius` around the z axis between -half and half.
fn ray_side(p: vec3<f32>, d: vec3<f32>, radius: f32, half: f32) -> vec4<f32> {
    let a = d.x * d.x + d.y * d.y;
    let b = p.x * d.x + p.y * d.y;
    let c = p.x * p.x + p.y * p.y - radius * radius;
    let disc = b * b - a * c;
    if a > 0.0 && disc >= 0.0 {
        let t = (-b - sqrt(disc)) / a;
        if t > 0.0 && abs(p.z + d.z * t) <= half {
            return vec4<f32>(vec3<f32>(p.x + d.x * t, p.y + d.y * t, 0.0) / radius, t);
        }
    }
    return vec4<f32>(0.0, 0.0, 1.0, NO_HIT);
}

fn nearer(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    if b.w > 0.0 && (a.w <= 0.0 || b.w < a.w) {
        return b;
    }
    return a;
}

fn ray_cylinder(p: vec3<f32>, d: vec3<f32>, radius: f32, half: f32) -> vec4<f32> {
    var best = ray_side(p, d, radius, half);
    for (var k = 0; k < 2; k++) {
        let side = select(-1.0, 1.0, k == 1);
        if d.z != 0.0 {
            let t = (side * half - p.z) / d.z;
            let x = p.x + d.x * t;
            let y = p.y + d.y * t;
            if t > 0.0 && x * x + y * y <= radius * radius {
                best = nearer(best, vec4<f32>(0.0, 0.0, side, t));
            }
        }
    }
    return best;
}

fn ray_capsule(p: vec3<f32>, d: vec3<f32>, radius: f32, half: f32) -> vec4<f32> {
    var best = ray_side(p, d, radius, half);
    for (var k = 0; k < 2; k++) {
        let end = vec3<f32>(0.0, 0.0, select(-half, half, k == 1));
        let t = ray_sphere(p, d, end, radius);
        if t > 0.0 {
            best = nearer(best, vec4<f32>((p + d * t - end) / radius, t));
        }
    }
    return best;
}

fn ray_grid(o: Obstacle, p: vec3<f32>, d: vec3<f32>, limit: f32) -> vec4<f32> {
    let voxel = o.half.w;
    let lo = o.half.xyz - voxel;
    let hi = o.half.xyz + (vec3<f32>(f32(o.nx), f32(o.ny), f32(o.nz)) - 1.0) * voxel + voxel;
    let inv = reciprocal(d);
    let t0 = (lo - p) * inv;
    let t1 = (hi - p) * inv;
    let near = min(t0, t1);
    let far = max(t0, t1);
    var t = max(max(max(near.x, near.y), near.z), 0.0);
    let end = min(min(min(far.x, far.y), far.z), limit);
    let length_d = length(d);
    for (var step = 0u; step < GRID_STEPS; step++) {
        if t > end {
            break;
        }
        let gd = grid_distance(o, p + d * t);
        if gd.w < GRID_HIT {
            let n = gd.xyz;
            let len = length(n);
            return vec4<f32>(select(vec3<f32>(0.0), n / len, len > 0.0), t);
        }
        t += gd.w / length_d;
    }
    return vec4<f32>(0.0, 0.0, 1.0, NO_HIT);
}

// (world normal, t) where the ray first meets `o` before `limit`, or t = NO_HIT.
fn intersect(o: Obstacle, origin: vec3<f32>, dir: vec3<f32>, limit: f32) -> vec4<f32> {
    let kind = u32(o.center.w);
    var hit: vec4<f32>;
    if kind == SPHERE {
        let t = ray_sphere(origin, dir, o.center.xyz, o.half.x);
        hit = vec4<f32>((origin + dir * t - o.center.xyz) / o.half.x, t);
    } else {
        let v = origin - o.center.xyz;
        let p = vec3<f32>(dot(o.r0.xyz, v), dot(o.r1.xyz, v), dot(o.r2.xyz, v));
        let d = vec3<f32>(dot(o.r0.xyz, dir), dot(o.r1.xyz, dir), dot(o.r2.xyz, dir));
        if kind == CUBOID {
            hit = ray_box(p, d, o.half.xyz);
        } else if kind == CYLINDER {
            hit = ray_cylinder(p, d, o.half.x, o.half.y);
        } else if kind == CAPSULE {
            hit = ray_capsule(p, d, o.half.x, o.half.y);
        } else {
            hit = ray_grid(o, p, d, limit);
        }
        hit = vec4<f32>(o.r0.xyz * hit.x + o.r1.xyz * hit.y + o.r2.xyz * hit.z, hit.w);
    }
    return select(vec4<f32>(0.0, 0.0, 1.0, NO_HIT), hit, hit.w > 0.0 && hit.w < limit);
}

@compute @workgroup_size(WORKGROUP)
fn render_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = gid.x + gid.y * groups.x * WORKGROUP;
    let per_view = R.width * R.height;
    if id >= R.views * per_view {
        return;
    }
    let index = id / per_view;
    let view = views[index];
    let pixel = id % per_view;
    let u = f32(pixel % R.width);
    let v = f32(pixel / R.width);
    let dir = view.eye0.xyz * ((u - R.cx) / R.fx) + view.eye1.xyz * ((v - R.cy) / R.fy) + view.eye2.xyz;
    let origin = view.eye.xyz;

    var best = FAR;
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    var color = ROBOT;
    for (var s = 0u; s < R.spheres; s++) {
        let sp = robot_spheres[index * R.spheres + s];
        let t = ray_sphere(origin, dir, sp.xyz, sp.w);
        if t > 0.0 && t < best {
            best = t;
            normal = (origin + dir * t - sp.xyz) / sp.w;
        }
    }
    let range = world_ranges[view.world];
    for (var k = 0u; k < range.y; k++) {
        var o = obstacles[range.x + k];
        if k == view.moved {
            o.center = vec4<f32>(view.moved_center.xyz, o.center.w);
            o.r0 = view.moved_r0;
            o.r1 = view.moved_r1;
            o.r2 = view.moved_r2;
        }
        let hit = intersect(o, origin, dir, best);
        if hit.w > 0.0 {
            best = hit.w;
            normal = hit.xyz;
            color = PALETTE[k % 8u];
        }
    }
    var depth = 0.0;
    if best < FAR {
        let light = normalize(vec3<f32>(0.3, 0.2, 1.0));
        color = color * (0.35 + 0.65 * max(dot(normal, light), 0.0));
        depth = best;
    } else {
        color = BACKGROUND;
    }
    let bytes = vec3<u32>(round(color * 255.0));
    colors[id] = bytes.x | (bytes.y << 8u) | (bytes.z << 16u);
    depths[id] = depth;
}
