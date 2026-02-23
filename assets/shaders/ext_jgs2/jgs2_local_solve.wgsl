// JGS2 per-vertex block solve: δx = -H⁻¹·g, q += δx
// Dispatch: ceil(node_count / 64) workgroups
//
// Reads accumulated gradient (3 floats) and Hessian diagonal (3×3).
// Phase 2: adds positive correction Δᵢ = Hᵢᵢ_rest - Sᵢ (regularizer).
// Makes effective Hessian larger → prevents overshoot (paper Eq. 15).
// Solves the 3×3 system analytically via cofactor inverse.

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> gradient: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> hessian_diag: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> q: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;
@group(0) @binding(5) var<storage, read> correction: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    // Skip pinned nodes
    if (mass[id].inv_mass <= 0.0) {
        return;
    }

    // Read gradient (bitcast from atomic u32)
    let base_g = id * 4u;
    let gx = bitcast<f32>(atomicLoad(&gradient[base_g + 0u]));
    let gy = bitcast<f32>(atomicLoad(&gradient[base_g + 1u]));
    let gz = bitcast<f32>(atomicLoad(&gradient[base_g + 2u]));

    // Read 3x3 Hessian (row-major, bitcast from atomic u32)
    let base_h = id * 9u;
    var H: array<f32, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        H[i] = bitcast<f32>(atomicLoad(&hessian_diag[base_h + i]));
    }

    // Phase 2: Add positive correction Δᵢ = Hᵢᵢ_rest - Sᵢ.
    // Δᵢ ≥ 0 (PSD), so H + Δ ≥ H. Always safe, no clamping needed.
    let base_c = id * 9u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        H[i] += correction[base_c + i];
    }

    // 3x3 inverse via cofactors (row-major)
    let a00 = H[0]; let a01 = H[1]; let a02 = H[2];
    let a10 = H[3]; let a11 = H[4]; let a12 = H[5];
    let a20 = H[6]; let a21 = H[7]; let a22 = H[8];

    let det = a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);

    // Guard: skip if determinant is too small or NaN
    if (abs(det) < 1e-12 || det != det) {
        return;
    }

    let inv_det = 1.0 / det;

    // Adjugate (cofactor transposed) entries
    let inv00 = (a11 * a22 - a12 * a21) * inv_det;
    let inv01 = (a02 * a21 - a01 * a22) * inv_det;
    let inv02 = (a01 * a12 - a02 * a11) * inv_det;
    let inv10 = (a12 * a20 - a10 * a22) * inv_det;
    let inv11 = (a00 * a22 - a02 * a20) * inv_det;
    let inv12 = (a02 * a10 - a00 * a12) * inv_det;
    let inv20 = (a10 * a21 - a11 * a20) * inv_det;
    let inv21 = (a01 * a20 - a00 * a21) * inv_det;
    let inv22 = (a00 * a11 - a01 * a10) * inv_det;

    // δx = -H⁻¹ · g
    let dx = -(inv00 * gx + inv01 * gy + inv02 * gz);
    let dy = -(inv10 * gx + inv11 * gy + inv12 * gz);
    let dz = -(inv20 * gx + inv21 * gy + inv22 * gz);

    // Update q
    let old_q = q[id];
    q[id] = vec4f(old_q.x + dx, old_q.y + dy, old_q.z + dz, 1.0);
}
