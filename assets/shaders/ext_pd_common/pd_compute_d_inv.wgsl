// Compute D_inv: inverse of each 3x3 diagonal block
// Uses adjugate/determinant method for 3x3 matrix inverse.
// If determinant is near zero (< 1e-20), writes identity matrix.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_inv: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let base = id * 9u;

    // Read 3x3 matrix (row-major layout)
    let a00 = diag[base + 0u]; let a01 = diag[base + 1u]; let a02 = diag[base + 2u];
    let a10 = diag[base + 3u]; let a11 = diag[base + 4u]; let a12 = diag[base + 5u];
    let a20 = diag[base + 6u]; let a21 = diag[base + 7u]; let a22 = diag[base + 8u];

    // Compute cofactors
    let cofactor00 = a11 * a22 - a12 * a21;
    let cofactor01 = -(a10 * a22 - a12 * a20);
    let cofactor02 = a10 * a21 - a11 * a20;
    let cofactor10 = -(a01 * a22 - a02 * a21);
    let cofactor11 = a00 * a22 - a02 * a20;
    let cofactor12 = -(a00 * a21 - a01 * a20);
    let cofactor20 = a01 * a12 - a02 * a11;
    let cofactor21 = -(a00 * a12 - a02 * a10);
    let cofactor22 = a00 * a11 - a01 * a10;

    // Determinant via first row expansion
    let det = a00 * cofactor00 + a01 * cofactor01 + a02 * cofactor02;

    if (abs(det) < 1e-20) {
        // Singular or near-singular: write identity matrix
        d_inv[base + 0u] = 1.0; d_inv[base + 1u] = 0.0; d_inv[base + 2u] = 0.0;
        d_inv[base + 3u] = 0.0; d_inv[base + 4u] = 1.0; d_inv[base + 5u] = 0.0;
        d_inv[base + 6u] = 0.0; d_inv[base + 7u] = 0.0; d_inv[base + 8u] = 1.0;
    } else {
        // Inverse = adjugate^T / det (cofactor matrix transposed)
        let inv_det = 1.0 / det;
        d_inv[base + 0u] = cofactor00 * inv_det;
        d_inv[base + 1u] = cofactor10 * inv_det;
        d_inv[base + 2u] = cofactor20 * inv_det;
        d_inv[base + 3u] = cofactor01 * inv_det;
        d_inv[base + 4u] = cofactor11 * inv_det;
        d_inv[base + 5u] = cofactor21 * inv_det;
        d_inv[base + 6u] = cofactor02 * inv_det;
        d_inv[base + 7u] = cofactor12 * inv_det;
        d_inv[base + 8u] = cofactor22 * inv_det;
    }
}
