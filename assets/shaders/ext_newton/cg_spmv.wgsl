// Generic CSR sparse matrix-vector product: Ap = A * p
// where A is stored as diagonal 3x3 blocks + off-diagonal CSR 3x3 blocks.
//
// All terms have already pre-multiplied their contributions into A:
//   InertialTerm: diag += M * I3x3
//   SpringTerm:   diag += dt^2 * H_diag,  offdiag += -dt^2 * H_offdiag
//
// No physics-specific knowledge — pure linear algebra.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> cg_p: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_ap: array<vec4f>;
@group(0) @binding(3) var<storage, read> csr_row_ptr: array<u32>;
@group(0) @binding(4) var<storage, read> csr_col_idx: array<u32>;
@group(0) @binding(5) var<storage, read> csr_values: array<f32>;
@group(0) @binding(6) var<storage, read> diag: array<f32>;

fn read_csr_block(offset: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(csr_values[offset + 0u], csr_values[offset + 1u], csr_values[offset + 2u]),
        vec3f(csr_values[offset + 3u], csr_values[offset + 4u], csr_values[offset + 5u]),
        vec3f(csr_values[offset + 6u], csr_values[offset + 7u], csr_values[offset + 8u]),
    );
}

fn read_diag_block(node: u32) -> mat3x3f {
    let base = node * 9u;
    return mat3x3f(
        vec3f(diag[base + 0u], diag[base + 1u], diag[base + 2u]),
        vec3f(diag[base + 3u], diag[base + 4u], diag[base + 5u]),
        vec3f(diag[base + 6u], diag[base + 7u], diag[base + 8u]),
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

    let pi = cg_p[id].xyz;

    // Diagonal: A_ii * p_i
    let d = read_diag_block(id);
    var result = mat3_mul_vec3(d, pi);

    // Off-diagonal: sum_j A_ij * p_j
    let row_start = csr_row_ptr[id];
    let row_end = csr_row_ptr[id + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col = csr_col_idx[idx];
        let block = read_csr_block(idx * 9u);
        let pj = cg_p[col].xyz;
        result = result + mat3_mul_vec3(block, pj);
    }

    cg_ap[id] = vec4f(result, 0.0);
}
