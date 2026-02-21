// FEM/SVD Area Preservation Constraint
// Dispatch: ceil(face_count / 64) workgroups
//
// Energy: E = A0 * 0.5 * k * (J - 1)^2
// where J = sigma1 * sigma2 is the area ratio from the SVD of the
// deformation gradient F = Ds * Dm_inv.
//
// Forces: PK1 stress via singular value decomposition
// Hessian: SVD-projected PSD (Teran 2005 / Smith 2019)

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
    stiffness: f32,        // area preservation (bulk modulus k)
    shear_stiffness: f32,  // trace energy / shear modulus (μ)
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

@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> triangles: array<AreaTriangle>;
@group(0) @binding(4) var<storage, read_write> diag_values: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> area_params: AreaParams;
@group(0) @binding(6) var<storage, read_write> csr_values: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> face_csr_map: array<FaceCSRMapping>;

// Accumulate a 3x3 block = scale * a * b^T into off-diagonal CSR
fn atomicAddOuter(base: u32, a: vec3f, b: vec3f, s: f32) {
    atomicAddFloat(&csr_values[base + 0u], s * a.x * b.x);
    atomicAddFloat(&csr_values[base + 1u], s * a.x * b.y);
    atomicAddFloat(&csr_values[base + 2u], s * a.x * b.z);
    atomicAddFloat(&csr_values[base + 3u], s * a.y * b.x);
    atomicAddFloat(&csr_values[base + 4u], s * a.y * b.y);
    atomicAddFloat(&csr_values[base + 5u], s * a.y * b.z);
    atomicAddFloat(&csr_values[base + 6u], s * a.z * b.x);
    atomicAddFloat(&csr_values[base + 7u], s * a.z * b.y);
    atomicAddFloat(&csr_values[base + 8u], s * a.z * b.z);
}

// Accumulate a 3x3 block = scale * a * b^T into diagonal buffer
fn atomicAddDiagOuter(base: u32, a: vec3f, b: vec3f, s: f32) {
    atomicAddFloat(&diag_values[base + 0u], s * a.x * b.x);
    atomicAddFloat(&diag_values[base + 1u], s * a.x * b.y);
    atomicAddFloat(&diag_values[base + 2u], s * a.x * b.z);
    atomicAddFloat(&diag_values[base + 3u], s * a.y * b.x);
    atomicAddFloat(&diag_values[base + 4u], s * a.y * b.y);
    atomicAddFloat(&diag_values[base + 5u], s * a.y * b.z);
    atomicAddFloat(&diag_values[base + 6u], s * a.z * b.x);
    atomicAddFloat(&diag_values[base + 7u], s * a.z * b.y);
    atomicAddFloat(&diag_values[base + 8u], s * a.z * b.z);
}

// Compute and accumulate a 3x3 Hessian block for vertex pair (i,j).
// Uses precomputed SVD basis (u1,u2,u3), Hessian coefficients, and
// the 2D weight vectors wi = (dot(ci,v1), dot(ci,v2)).
fn accumulateBlock(
    base: u32,      // buffer base offset (node*9 for diag, csr_idx*9 for off-diag)
    is_diag: bool,  // true → write to diag_values, false → write to csr_values
    wi: vec2f,      // weight for vertex i
    wj: vec2f,      // weight for vertex j
    u1: vec3f, u2: vec3f, u3: vec3f,
    Q00: f32, Q01: f32, Q11: f32,
    half_tw_fl_sum: f32,  // (lam_twist + lam_flip) / 2
    half_tw_fl_diff: f32, // (lam_flip - lam_twist) / 2
    ln1: f32, ln2: f32,   // null-space coefficients
    scale: f32
) {
    let A1 = wi.x * wj.x;
    let A2 = wi.x * wj.y;
    let A3 = wi.y * wj.x;
    let A4 = wi.y * wj.y;

    // Combined coefficients for each rank-1 basis
    let c11 = Q00 * A1 + half_tw_fl_sum * A4;
    let c12 = Q01 * A2 + half_tw_fl_diff * A3;
    let c21 = Q01 * A3 + half_tw_fl_diff * A2;
    let c22 = Q11 * A4 + half_tw_fl_sum * A1;
    let c33 = ln1 * A1 + ln2 * A4;

    // H = scale * (c11*u1*u1^T + c12*u1*u2^T + c21*u2*u1^T + c22*u2*u2^T + c33*u3*u3^T)
    // Compute 3 row vectors, then write 9 entries
    let row1 = c11 * u1 + c12 * u2;
    let row2 = c21 * u1 + c22 * u2;
    let row3 = c33 * u3;

    // For each spatial row m, the 3D vector is:
    //   H[m,:] = scale * (u1[m]*row1 + u2[m]*row2 + u3[m]*row3)
    let hx = scale * (u1.x * row1 + u2.x * row2 + u3.x * row3);
    let hy = scale * (u1.y * row1 + u2.y * row2 + u3.y * row3);
    let hz = scale * (u1.z * row1 + u2.z * row2 + u3.z * row3);

    if (is_diag) {
        atomicAddFloat(&diag_values[base + 0u], hx.x);
        atomicAddFloat(&diag_values[base + 1u], hx.y);
        atomicAddFloat(&diag_values[base + 2u], hx.z);
        atomicAddFloat(&diag_values[base + 3u], hy.x);
        atomicAddFloat(&diag_values[base + 4u], hy.y);
        atomicAddFloat(&diag_values[base + 5u], hy.z);
        atomicAddFloat(&diag_values[base + 6u], hz.x);
        atomicAddFloat(&diag_values[base + 7u], hz.y);
        atomicAddFloat(&diag_values[base + 8u], hz.z);
    } else {
        atomicAddFloat(&csr_values[base + 0u], hx.x);
        atomicAddFloat(&csr_values[base + 1u], hx.y);
        atomicAddFloat(&csr_values[base + 2u], hx.z);
        atomicAddFloat(&csr_values[base + 3u], hy.x);
        atomicAddFloat(&csr_values[base + 4u], hy.y);
        atomicAddFloat(&csr_values[base + 5u], hy.z);
        atomicAddFloat(&csr_values[base + 6u], hz.x);
        atomicAddFloat(&csr_values[base + 7u], hz.y);
        atomicAddFloat(&csr_values[base + 8u], hz.z);
    }
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let fid = gid.x;
    if (fid >= params.face_count) {
        return;
    }

    let tri = triangles[fid];
    let na = tri.n0;
    let nb = tri.n1;
    let nc = tri.n2;
    let A0 = tri.rest_area;
    let k = area_params.stiffness;

    // Current positions
    let x0 = positions[na].xyz;
    let x1 = positions[nb].xyz;
    let x2 = positions[nc].xyz;

    // Deformed edges (3D)
    let ds0 = x1 - x0;
    let ds1 = x2 - x0;

    // Deformation gradient F = Ds * Dm_inv (3x2, stored as two column vec3s)
    let dm00 = tri.dm_inv_00;
    let dm01 = tri.dm_inv_01;
    let dm10 = tri.dm_inv_10;
    let dm11 = tri.dm_inv_11;
    let f0 = ds0 * dm00 + ds1 * dm10;  // F column 0
    let f1 = ds0 * dm01 + ds1 * dm11;  // F column 1

    // Cauchy-Green C = F^T * F (2x2 symmetric)
    let c00 = dot(f0, f0);
    let c01 = dot(f0, f1);
    let c11 = dot(f1, f1);
    let det_C = c00 * c11 - c01 * c01;

    if (det_C < 1e-20) {
        return;  // degenerate triangle
    }

    // ===== SVD via eigendecomposition of C =====
    let half_sum = 0.5 * (c00 + c11);
    let half_diff = 0.5 * (c00 - c11);
    let disc = sqrt(half_diff * half_diff + c01 * c01);
    let lam1 = max(half_sum + disc, 1e-12);  // larger eigenvalue
    let lam2 = max(half_sum - disc, 1e-12);  // smaller eigenvalue

    let sig1 = sqrt(lam1);
    let sig2 = sqrt(lam2);
    let J = sig1 * sig2;  // area ratio

    // Eigenvectors of C -> V rotation matrix
    // Guard atan2(0,0): when C is diagonal (c01≈0, c00≈c11), any rotation is valid.
    // Some GPU drivers return NaN for atan2(0,0), which propagates via 0*NaN=NaN.
    let atan_y = 2.0 * c01;
    let atan_x = c00 - c11;
    var theta = 0.0;
    if (abs(atan_y) > 1e-20 || abs(atan_x) > 1e-20) {
        theta = 0.5 * atan2(atan_y, atan_x);
    }
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let v1 = vec2f(cos_t, sin_t);
    let v2 = vec2f(-sin_t, cos_t);

    // Left singular vectors: U = F * V * Sigma^{-1}
    let u1 = (f0 * v1.x + f1 * v1.y) / sig1;
    let u2 = (f0 * v2.x + f1 * v2.y) / sig2;
    let u3 = cross(u1, u2);  // triangle normal direction

    // ===== Forces via PK1 stress =====
    let mu = area_params.shear_stiffness;  // trace/Dirichlet energy shear modulus

    // Combined area + ARAP shear energy:
    //   psi = 0.5*k*(J-1)^2 + mu/2*((sigma1-1)^2 + (sigma2-1)^2)
    // ARAP forces are zero at rest (sigma=1), unlike trace energy.
    let Jm1 = J - 1.0;
    let p1 = k * Jm1 * sig2 + mu * (sig1 - 1.0);
    let p2 = k * Jm1 * sig1 + mu * (sig2 - 1.0);

    // P = U * diag(p1,p2) * V^T  (3x2 PK1 stress)
    let P0 = u1 * (p1 * v1.x) + u2 * (p2 * v2.x);  // column 0
    let P1 = u1 * (p1 * v1.y) + u2 * (p2 * v2.y);  // column 1

    // Vertex forces = -A0 * P * Dm_inv^T
    // ci vectors (rows of Dm_inv for each vertex)
    let ci1 = vec2f(dm00, dm01);  // row 0 -> vertex 1 (n1)
    let ci2 = vec2f(dm10, dm11);  // row 1 -> vertex 2 (n2)
    let ci0 = -(ci1 + ci2);      // vertex 0 (n0)

    let force0 = -A0 * (P0 * ci0.x + P1 * ci0.y);
    let force1 = -A0 * (P0 * ci1.x + P1 * ci1.y);
    let force2 = -A0 * (P0 * ci2.x + P1 * ci2.y);

    // Accumulate forces
    let ba = na * 4u;
    atomicAddFloat(&forces[ba + 0u], force0.x);
    atomicAddFloat(&forces[ba + 1u], force0.y);
    atomicAddFloat(&forces[ba + 2u], force0.z);

    let bb = nb * 4u;
    atomicAddFloat(&forces[bb + 0u], force1.x);
    atomicAddFloat(&forces[bb + 1u], force1.y);
    atomicAddFloat(&forces[bb + 2u], force1.z);

    let bc = nc * 4u;
    atomicAddFloat(&forces[bc + 0u], force2.x);
    atomicAddFloat(&forces[bc + 1u], force2.y);
    atomicAddFloat(&forces[bc + 2u], force2.z);

    // ===== SVD-Projected PSD Hessian =====
    let dt2 = params.dt * params.dt;
    let scale = dt2 * A0;

    // Stretch Hessian in singular value space (2x2)
    // Combined area + trace: d²ψ/dσᵢdσⱼ includes +μ on diagonal
    let h11 = k * sig2 * sig2 + mu;
    let h22 = k * sig1 * sig1 + mu;
    let h12 = k * (2.0 * J - 1.0);

    // PSD clamp: eigendecompose the 2x2 stretch Hessian
    let s_half_sum = 0.5 * (h11 + h22);
    let s_half_diff = 0.5 * (h11 - h22);
    let s_disc = sqrt(s_half_diff * s_half_diff + h12 * h12);
    let s_lam1 = max(s_half_sum + s_disc, 0.0);
    let s_lam2 = max(s_half_sum - s_disc, 0.0);

    // Reconstruct clamped Q = R * diag(s_lam) * R^T
    // Guard atan2(0,0) — same GPU driver issue as SVD theta above.
    let s_atan_y = 2.0 * h12;
    let s_atan_x = h11 - h22;
    var s_theta = 0.0;
    if (abs(s_atan_y) > 1e-20 || abs(s_atan_x) > 1e-20) {
        s_theta = 0.5 * atan2(s_atan_y, s_atan_x);
    }
    let sc = cos(s_theta);
    let ss = sin(s_theta);
    let Q00 = sc * sc * s_lam1 + ss * ss * s_lam2;
    let Q01 = sc * ss * (s_lam1 - s_lam2);
    let Q11 = ss * ss * s_lam1 + sc * sc * s_lam2;

    // Twist: (p1-p2)/(sig1-sig2) = -k*(J-1) + μ
    // Flip:  (p1+p2)/(sig1+sig2) = k*(J-1) + μ  (trace energy approximation)
    let lam_twist = max(-k * Jm1 + mu, 0.0);
    let lam_flip = max(k * Jm1 + mu, 0.0);

    // Null-space (u3 direction): p_i/sigma_i → 0 at rest.
    // No artificial floor — area energy has no bending stiffness.
    let lam_n1 = select(0.0, max(p1 / sig1, 0.0), sig1 > 1e-8);
    let lam_n2 = select(0.0, max(p2 / sig2, 0.0), sig2 > 1e-8);

    // Precompute combined twist/flip terms
    let a_coeff = 0.5 * (lam_twist + lam_flip);
    let b_coeff = 0.5 * (lam_flip - lam_twist);

    // Weight vectors: wi = (dot(ci, v1), dot(ci, v2))
    let w0 = vec2f(dot(ci0, v1), dot(ci0, v2));
    let w1 = vec2f(dot(ci1, v1), dot(ci1, v2));
    let w2 = vec2f(dot(ci2, v1), dot(ci2, v2));

    let mapping = face_csr_map[fid];

    // Diagonal blocks (i == j)
    accumulateBlock(na * 9u, true, w0, w0, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(nb * 9u, true, w1, w1, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(nc * 9u, true, w2, w2, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);

    // Off-diagonal blocks
    accumulateBlock(mapping.csr_01 * 9u, false, w0, w1, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(mapping.csr_10 * 9u, false, w1, w0, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(mapping.csr_02 * 9u, false, w0, w2, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(mapping.csr_20 * 9u, false, w2, w0, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(mapping.csr_12 * 9u, false, w1, w2, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
    accumulateBlock(mapping.csr_21 * 9u, false, w2, w1, u1, u2, u3, Q00, Q01, Q11, a_coeff, b_coeff, lam_n1, lam_n2, scale);
}
