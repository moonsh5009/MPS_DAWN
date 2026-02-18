// Assemble RHS vector: b = dt * F - M * dv_total
// Dispatch: ceil(node_count / 64) workgroups
//
// Newton outer loop: the RHS uses accumulated velocity delta (dv_total)
// instead of CSR matrix-vector product against velocity.
// Forces are f32 (accumulated via CAS-based atomic float add).

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
@group(0) @binding(1) var<storage, read> forces: array<f32>;
@group(0) @binding(2) var<storage, read> dv_total: array<vec4f>;
@group(0) @binding(3) var<storage, read> mass: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rhs: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    // MPCG filter: zero RHS for pinned (infinite mass) nodes
    let inv_mass = mass[id].y;
    if (inv_mass <= 0.0) {
        rhs[id] = vec4f(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Read force directly as f32
    let f_base = id * 4u;
    let f = vec3f(forces[f_base + 0u], forces[f_base + 1u], forces[f_base + 2u]);

    let m = mass[id].x;
    let dv = dv_total[id].xyz;

    // b = dt * F - M * dv_total
    let b = params.dt * f - m * dv;

    rhs[id] = vec4f(b, 0.0);
}
