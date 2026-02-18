// CG direction update: p = r + beta * p
// Dispatch: ceil(node_count / 64) workgroups
//
// Reads beta from scalars[4] (computed by cg_compute_scalars).

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
@group(0) @binding(1) var<storage, read> cg_r: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_p: array<vec4f>;
@group(0) @binding(3) var<storage, read> scalars: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let beta = scalars[4];

    let r = cg_r[id].xyz;
    let p = cg_p[id].xyz;

    cg_p[id] = vec4f(r + beta * p, 0.0);

    // MPCG filter: zero search direction for pinned (infinite mass) nodes
    let inv_mass = mass[id].y;
    if (inv_mass <= 0.0) {
        cg_p[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
