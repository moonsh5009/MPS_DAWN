// JGS2 spring gradient + diagonal Hessian accumulation
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a, b) with stiffness k and rest_length L:
//   dx = q[a] - q[b], dist = |dx|, dir = dx/dist
//   Gradient: g_a += k*(dist-L)*dir, g_b -= k*(dist-L)*dir
//   Diagonal Hessian (PSD projected):
//     H = k*((1-min(L/d,1))*I + min(L/d,1)*dir⊗dir)
//     H_aa += H, H_bb += H
//
// Uses atomicAddFloat for thread-safe accumulation (multiple edges per vertex).

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

struct SpringParams {
    stiffness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> q: array<vec4f>;
@group(0) @binding(2) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(3) var<uniform> spring_params: SpringParams;
@group(0) @binding(4) var<storage, read_write> gradient: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> hessian_diag: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= solver.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;
    let k = spring_params.stiffness;
    let rest_len = edge.rest_length;

    let pa = q[a].xyz;
    let pb = q[b].xyz;
    let dx = pa - pb;
    let dist = length(dx);

    if (dist < 1e-8) {
        return;
    }

    let dir = dx / dist;

    // --- Gradient (energy gradient, not force) ---
    let g_spring = k * (dist - rest_len) * dir;

    let base_a = a * 4u;
    atomicAddFloat(&gradient[base_a + 0u], g_spring.x);
    atomicAddFloat(&gradient[base_a + 1u], g_spring.y);
    atomicAddFloat(&gradient[base_a + 2u], g_spring.z);

    let base_b = b * 4u;
    atomicAddFloat(&gradient[base_b + 0u], -g_spring.x);
    atomicAddFloat(&gradient[base_b + 1u], -g_spring.y);
    atomicAddFloat(&gradient[base_b + 2u], -g_spring.z);

    // --- Diagonal Hessian (PSD projected) ---
    let ratio = min(rest_len / dist, 1.0);
    let coeff_i = k * (1.0 - ratio);
    let coeff_d = k * ratio;

    // 3x3 block H (row-major)
    var h: array<f32, 9>;
    h[0] = coeff_i + coeff_d * dir.x * dir.x;
    h[1] = coeff_d * dir.x * dir.y;
    h[2] = coeff_d * dir.x * dir.z;
    h[3] = coeff_d * dir.y * dir.x;
    h[4] = coeff_i + coeff_d * dir.y * dir.y;
    h[5] = coeff_d * dir.y * dir.z;
    h[6] = coeff_d * dir.z * dir.x;
    h[7] = coeff_d * dir.z * dir.y;
    h[8] = coeff_i + coeff_d * dir.z * dir.z;

    // Accumulate diagonal: H_aa += H, H_bb += H
    let diag_a = a * 9u;
    let diag_b = b * 9u;

    for (var i = 0u; i < 9u; i = i + 1u) {
        atomicAddFloat(&hessian_diag[diag_a + i], h[i]);
        atomicAddFloat(&hessian_diag[diag_b + i], h[i]);
    }
}
