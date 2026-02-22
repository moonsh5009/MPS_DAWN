// PD Area LHS: Constant system matrix assembly (w * S^T * S for triangles)
// Dispatch: ceil(face_count / 64) workgroups
//
// For each triangle (n0,n1,n2) with weight w = stiffness * rest_area:
//   The selection matrix S maps vertex positions to deformation gradient
//   via Dm_inv coefficients:
//     coeff_0j = -(dm_inv[0,j] + dm_inv[1,j])   (node 0, dependent)
//     coeff_1j = dm_inv[0,j]                     (node 1)
//     coeff_2j = dm_inv[1,j]                     (node 2)
//
//   c_ab = coeff_a0 * coeff_b0 + coeff_a1 * coeff_b1  (2D dot product)
//
//   Diagonal blocks:     diag[n_alpha]           += c_aa * w * I_3
//   Off-diagonal blocks: csr[n_alpha, n_beta]    += c_ab * w * I_3
//
// Only diagonal entries [0],[4],[8] of each 3x3 block are written
// since all blocks are scalar multiples of the identity matrix.

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

struct AreaTriangle {
    n0: u32,
    n1: u32,
    n2: u32,
    rest_area: f32,
    dm_inv_00: f32,
    dm_inv_01: f32,
    dm_inv_10: f32,
    dm_inv_11: f32,
};

struct AreaParams {
    stiffness: f32,
    shear_stiffness: f32,
    _pad1: f32,
    _pad2: f32,
};

struct FaceCSRMapping {
    csr_01: u32,
    csr_10: u32,
    csr_02: u32,
    csr_20: u32,
    csr_12: u32,
    csr_21: u32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> triangles: array<AreaTriangle>;
@group(0) @binding(2) var<storage, read_write> diag: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> csr_values: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> face_csr_map: array<FaceCSRMapping>;
@group(0) @binding(5) var<uniform> area_params: AreaParams;

// Accumulate c * w to diagonal entries [0],[4],[8] of a 3x3 block
fn atomicAddDiagScalar(node: u32, val: f32) {
    let base = node * 9u;
    atomicAddFloat(&diag[base + 0u], val);
    atomicAddFloat(&diag[base + 4u], val);
    atomicAddFloat(&diag[base + 8u], val);
}

// Accumulate c * w to diagonal entries [0],[4],[8] of a CSR 3x3 block
fn atomicAddCSRScalar(csr_idx: u32, val: f32) {
    let base = csr_idx * 9u;
    atomicAddFloat(&csr_values[base + 0u], val);
    atomicAddFloat(&csr_values[base + 4u], val);
    atomicAddFloat(&csr_values[base + 8u], val);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let fid = gid.x;
    if (fid >= solver.face_count) {
        return;
    }

    let tri = triangles[fid];
    let na = tri.n0;
    let nb = tri.n1;
    let nc = tri.n2;
    let A0 = tri.rest_area;
    let k = area_params.stiffness;
    let w = k * A0;

    // Dm_inv-based coefficient vectors for each vertex (2D)
    let ci1 = vec2f(tri.dm_inv_00, tri.dm_inv_01);  // vertex 1 (n1)
    let ci2 = vec2f(tri.dm_inv_10, tri.dm_inv_11);  // vertex 2 (n2)
    let ci0 = -(ci1 + ci2);                          // vertex 0 (n0, dependent)

    // c_ab = dot(ci_a, ci_b) — 2D dot product of coefficient vectors
    let c00 = dot(ci0, ci0);
    let c11 = dot(ci1, ci1);
    let c22 = dot(ci2, ci2);
    let c01 = dot(ci0, ci1);
    let c02 = dot(ci0, ci2);
    let c12 = dot(ci1, ci2);

    // Diagonal blocks: diag[n_alpha] += c_aa * w * I_3
    atomicAddDiagScalar(na, c00 * w);
    atomicAddDiagScalar(nb, c11 * w);
    atomicAddDiagScalar(nc, c22 * w);

    // Off-diagonal blocks: csr[n_alpha, n_beta] += c_ab * w * I_3
    let mapping = face_csr_map[fid];

    atomicAddCSRScalar(mapping.csr_01, c01 * w);
    atomicAddCSRScalar(mapping.csr_10, c01 * w);
    atomicAddCSRScalar(mapping.csr_02, c02 * w);
    atomicAddCSRScalar(mapping.csr_20, c02 * w);
    atomicAddCSRScalar(mapping.csr_12, c12 * w);
    atomicAddCSRScalar(mapping.csr_21, c12 * w);
}
