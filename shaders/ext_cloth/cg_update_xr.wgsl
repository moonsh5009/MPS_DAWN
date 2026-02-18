// CG update: x += alpha * p, r -= alpha * Ap
// Dispatch: ceil(node_count / 64) workgroups
//
// Reads alpha from scalars[3] (computed by cg_compute_scalars).

struct ClothSimParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    node_count: u32,
    edge_count: u32,
    face_count: u32,
    cg_max_iter: u32,
    damping: f32,
    cg_tolerance: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> params: ClothSimParams;
@group(0) @binding(1) var<storage, read_write> cg_x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_r: array<vec4f>;
@group(0) @binding(3) var<storage, read> cg_p: array<vec4f>;
@group(0) @binding(4) var<storage, read> cg_ap: array<vec4f>;
@group(0) @binding(5) var<storage, read> scalars: array<f32>;
@group(0) @binding(6) var<storage, read> mass: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let alpha = scalars[3];

    let x = cg_x[id].xyz;
    let r = cg_r[id].xyz;
    let p = cg_p[id].xyz;
    let ap = cg_ap[id].xyz;

    cg_x[id] = vec4f(x + alpha * p, 0.0);
    cg_r[id] = vec4f(r - alpha * ap, 0.0);

    // MPCG filter: zero residual for pinned (infinite mass) nodes
    let inv_mass = mass[id].y;
    if (inv_mass <= 0.0) {
        cg_r[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
