// Accumulate spring forces and assemble Hessian blocks
// Dispatch: ceil(edge_count / 64) workgroups
//
// Per edge (a, b) with stiffness k and rest length L:
//   dx = pos[a] - pos[b], dist = |dx|, dir = dx / dist
//   Force:  f = -k * (dist - L) * dir
//   Cross Jacobian dfa/dxb (off-diagonal block):
//     J_ab = k * ((1 - L/dist) * I + (L/dist) * dir * dirT)
//   Self Jacobian dfa/dxa = -J_ab
//   Diagonal accumulation: diag_a += -J_ab = dfa/dxa
//
// The full Jacobian df/dx is negative semi-definite, ensuring
// the system matrix A = M - dt^2*J is always positive definite.
//
// edge_csr_mapping[e] = vec4u(csr_idx_ab, csr_idx_ba, diag_a, diag_b)

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

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

@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(5) var<storage, read_write> csr_values: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> diag_values: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> edge_csr_map: array<vec4u>;
@group(0) @binding(8) var<uniform> spring_params: SpringParams;

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

    let pa = positions[a].xyz;
    let pb = positions[b].xyz;
    let dx = pa - pb;
    let dist = length(dx);

    // Avoid division by zero for degenerate edges
    if (dist < 1e-8) {
        return;
    }

    let dir = dx / dist;

    // --- Spring force ---
    // f = -k * (dist - rest_len) * dir
    let f_spring = -k * (dist - rest_len) * dir;

    // Atomic add force to node a: force[a] += f_spring
    let base_a = a * 4u;
    atomicAddFloat(&forces[base_a + 0u], f_spring.x);
    atomicAddFloat(&forces[base_a + 1u], f_spring.y);
    atomicAddFloat(&forces[base_a + 2u], f_spring.z);

    // Atomic add force to node b: force[b] -= f_spring
    let base_b = b * 4u;
    atomicAddFloat(&forces[base_b + 0u], -f_spring.x);
    atomicAddFloat(&forces[base_b + 1u], -f_spring.y);
    atomicAddFloat(&forces[base_b + 2u], -f_spring.z);

    // --- Hessian assembly ---
    // Clamp ratio to [0, 1] so that coeff_i >= 0, ensuring the Jacobian block
    // is always positive semi-definite. Without this clamp, compressed edges
    // (dist < rest_len) produce negative tangential stiffness, making the
    // system matrix A = M - dt^2*J indefinite and causing CG to diverge → NaN.
    let ratio = min(rest_len / dist, 1.0);
    let coeff_i = k * (1.0 - ratio);    // coefficient for identity part (>= 0)
    let coeff_d = k * ratio;            // coefficient for dir*dirT part

    // 3x3 block H_ab (row-major)
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

    // Pre-multiply by dt² for generic SpMV (Ap = diag*p + offdiag*p).
    // System matrix A = M - dt²*J, so:
    //   offdiag A_ab = -dt²*J_ab = -dt²*h
    //   diagonal contribution = +dt²*J_ab = +dt²*h (since self-Jacobian = -J_ab)
    let dt2 = physics.dt_sq;

    // Write off-diagonal CSR blocks (symmetric: H_ab = H_ba)
    // Uses atomicAddFloat to accumulate with other terms (e.g. area) sharing CSR entries.
    let mapping = edge_csr_map[eid];
    let csr_ab = mapping.x * 9u;
    let csr_ba = mapping.y * 9u;

    for (var i = 0u; i < 9u; i = i + 1u) {
        atomicAddFloat(&csr_values[csr_ab + i], -dt2 * h[i]);
        atomicAddFloat(&csr_values[csr_ba + i], -dt2 * h[i]);
    }

    // Accumulate diagonal blocks: diag += dt²*H_ab per neighbor
    let diag_a = a * 9u;
    let diag_b = b * 9u;

    for (var i = 0u; i < 9u; i = i + 1u) {
        atomicAddFloat(&diag_values[diag_a + i], dt2 * h[i]);
        atomicAddFloat(&diag_values[diag_b + i], dt2 * h[i]);
    }
}
