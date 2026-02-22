// Fused off-diagonal SpMV + Jacobi + Chebyshev step for Projective Dynamics.
// Combines: offdiag = (A-D)*q_curr, z = D⁻¹*(b-offdiag), q_new = ω*(z-q_prev)+q_prev
// Eliminates the intermediate temp buffer.
// Dispatch: ceil(node_count / 64) workgroups

struct JacobiParams {
    omega: f32,
    is_first_step: u32,
    _pad0: f32,
    _pad1: f32,
};

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> q_curr: array<vec4f>;
@group(0) @binding(2) var<storage, read> csr_row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> csr_col_idx: array<u32>;
@group(0) @binding(4) var<storage, read> csr_values: array<f32>;
@group(0) @binding(5) var<storage, read> rhs: array<vec4f>;
@group(0) @binding(6) var<storage, read> d_inv: array<f32>;
@group(0) @binding(7) var<storage, read> q_prev: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> q_new: array<vec4f>;
@group(0) @binding(9) var<uniform> jacobi_params: JacobiParams;
@group(0) @binding(10) var<storage, read> mass: array<SimMass>;

fn read_csr_block(offset: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(csr_values[offset + 0u], csr_values[offset + 1u], csr_values[offset + 2u]),
        vec3f(csr_values[offset + 3u], csr_values[offset + 4u], csr_values[offset + 5u]),
        vec3f(csr_values[offset + 6u], csr_values[offset + 7u], csr_values[offset + 8u]),
    );
}

fn read_dinv_block(node: u32) -> mat3x3f {
    let base = node * 9u;
    return mat3x3f(
        vec3f(d_inv[base + 0u], d_inv[base + 1u], d_inv[base + 2u]),
        vec3f(d_inv[base + 3u], d_inv[base + 4u], d_inv[base + 5u]),
        vec3f(d_inv[base + 6u], d_inv[base + 7u], d_inv[base + 8u]),
    );
}

fn mat3_mul_vec3(m: mat3x3f, v: vec3f) -> vec3f {
    return vec3f(
        dot(m[0], v),
        dot(m[1], v),
        dot(m[2], v),
    );
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    // Pinned nodes: keep predicted position
    if (inv_mass <= 0.0) {
        q_new[id] = q_prev[id];
        return;
    }

    // Off-diagonal SpMV: offdiag = sum_{j!=i} A[i,j] * q_curr[j]
    var offdiag = vec3f(0.0);
    let row_start = csr_row_ptr[id];
    let row_end = csr_row_ptr[id + 1u];
    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col = csr_col_idx[idx];
        let block = read_csr_block(idx * 9u);
        offdiag = offdiag + mat3_mul_vec3(block, q_curr[col].xyz);
    }

    // Jacobi: z = D⁻¹ * (b - offdiag)
    let b = rhs[id].xyz;
    let d = read_dinv_block(id);
    let z = mat3_mul_vec3(d, b - offdiag);

    if (jacobi_params.is_first_step != 0u) {
        // First step: pure Jacobi (no Chebyshev blend)
        q_new[id] = vec4f(z, 1.0);
    } else {
        // Chebyshev: q_new = ω * (z - q_prev) + q_prev
        let qp = q_prev[id].xyz;
        let omega = jacobi_params.omega;
        q_new[id] = vec4f(omega * (z - qp) + qp, 1.0);
    }
}
