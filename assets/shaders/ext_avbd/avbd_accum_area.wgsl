// AVBD area accumulation: SVD-based FEM area term (vertex-centric gather).
// Each thread processes one vertex, loops over incident faces via face CSR,
// computes deformation gradient / SVD / PK1 stress / PSD Hessian diagonal block.
//
// Key differences from Newton area shader:
//   - Gradient sign: VBD gradient = dE/dx (positive), Newton force = -dE/dx (negative)
//   - Hessian scale: no dt² multiplier (inertia already scaled by mass*inv_dt²)
//   - Only diagonal Hessian block (no off-diagonal CSR needed)
//
// Dispatch: ceil(color_vertex_count / 64) per color group.

#import "ext_avbd/header/vbd_params.wgsl"

struct AreaParams {
    stiffness: f32,
    shear_stiffness: f32,
    _pad1: f32,
    _pad2: f32,
};

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

struct FaceAdjacency {
    face_idx: u32,
    vertex_role: u32,
};

@group(0) @binding(0) var<uniform> color_params: VBDColorParams;
@group(0) @binding(1) var<uniform> area_params: AreaParams;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> gradient: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> hessian: array<vec4f>;
@group(0) @binding(5) var<storage, read> vertex_order: array<u32>;
@group(0) @binding(6) var<storage, read> face_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> face_adjacency: array<FaceAdjacency>;
@group(0) @binding(8) var<storage, read> triangles: array<AreaTriangle>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let local_idx = gid.x;
    if (local_idx >= color_params.color_vertex_count) {
        return;
    }

    let vi = vertex_order[color_params.color_offset + local_idx];
    let k = area_params.stiffness;
    let mu = area_params.shear_stiffness;

    var gx = 0.0; var gy = 0.0; var gz = 0.0;
    var h00 = 0.0; var h01 = 0.0; var h02 = 0.0;
    var h10 = 0.0; var h11 = 0.0; var h12 = 0.0;
    var h20 = 0.0; var h21 = 0.0; var h22 = 0.0;

    let f_start = face_offsets[vi];
    let f_end = face_offsets[vi + 1u];

    for (var fi = f_start; fi < f_end; fi = fi + 1u) {
        let adj = face_adjacency[fi];
        let fid = adj.face_idx;
        let role = adj.vertex_role;

        let tri = triangles[fid];
        let x0 = q[tri.n0].xyz;
        let x1 = q[tri.n1].xyz;
        let x2 = q[tri.n2].xyz;
        let A0 = tri.rest_area;

        // Deformed edges
        let ds0 = x1 - x0;
        let ds1 = x2 - x0;

        // Deformation gradient F = Ds * Dm_inv (3×2)
        let dm00 = tri.dm_inv_00;
        let dm01 = tri.dm_inv_01;
        let dm10 = tri.dm_inv_10;
        let dm11 = tri.dm_inv_11;
        let f0 = ds0 * dm00 + ds1 * dm10;  // F column 0
        let f1 = ds0 * dm01 + ds1 * dm11;  // F column 1

        // Cauchy-Green C = F^T F (2×2 symmetric)
        let c00_v = dot(f0, f0);
        let c01_v = dot(f0, f1);
        let c11_v = dot(f1, f1);
        let det_C = c00_v * c11_v - c01_v * c01_v;

        if (det_C < 1e-20) {
            continue;  // degenerate triangle
        }

        // ===== SVD via eigendecomposition of C =====
        let half_sum = 0.5 * (c00_v + c11_v);
        let half_diff = 0.5 * (c00_v - c11_v);
        let disc = sqrt(half_diff * half_diff + c01_v * c01_v);
        let lam1 = max(half_sum + disc, 1e-12);
        let lam2 = max(half_sum - disc, 1e-12);

        let sig1 = sqrt(lam1);
        let sig2 = sqrt(lam2);
        let J = sig1 * sig2;

        // Eigenvectors of C → V rotation
        let atan_y = 2.0 * c01_v;
        let atan_x = c00_v - c11_v;
        var theta = 0.0;
        if (abs(atan_y) > 1e-20 || abs(atan_x) > 1e-20) {
            theta = 0.5 * atan2(atan_y, atan_x);
        }
        let cos_t = cos(theta);
        let sin_t = sin(theta);
        let v1 = vec2f(cos_t, sin_t);
        let v2 = vec2f(-sin_t, cos_t);

        let u1 = (f0 * v1.x + f1 * v1.y) / sig1;
        let u2 = (f0 * v2.x + f1 * v2.y) / sig2;
        let u3 = cross(u1, u2);

        // ===== PK1 stress =====
        let Jm1 = J - 1.0;
        let p1 = k * Jm1 * sig2 + mu * (sig1 - 1.0);
        let p2 = k * Jm1 * sig1 + mu * (sig2 - 1.0);

        // P = U * diag(p1,p2) * V^T (3×2)
        let P0 = u1 * (p1 * v1.x) + u2 * (p2 * v2.x);
        let P1 = u1 * (p1 * v1.y) + u2 * (p2 * v2.y);

        // ci vector selection based on vertex role
        let ci1 = vec2f(dm00, dm01);
        let ci2 = vec2f(dm10, dm11);
        var ci: vec2f;
        if (role == 0u) {
            ci = -(ci1 + ci2);
        } else if (role == 1u) {
            ci = ci1;
        } else {
            ci = ci2;
        }

        // Gradient: dE/dx_i = A0 * (P0 * ci.x + P1 * ci.y) (positive sign for VBD)
        let grad = A0 * (P0 * ci.x + P1 * ci.y);
        gx += grad.x;
        gy += grad.y;
        gz += grad.z;

        // ===== Hessian diagonal block (PSD-projected) =====
        // Stretch Hessian in SVD space (2×2)
        let h_s11 = k * sig2 * sig2 + mu;
        let h_s22 = k * sig1 * sig1 + mu;
        let h_s12 = k * (2.0 * J - 1.0);

        // PSD clamp: eigendecompose the 2×2 stretch Hessian
        let s_half_sum = 0.5 * (h_s11 + h_s22);
        let s_half_diff = 0.5 * (h_s11 - h_s22);
        let s_disc = sqrt(s_half_diff * s_half_diff + h_s12 * h_s12);
        let s_lam1 = max(s_half_sum + s_disc, 0.0);
        let s_lam2 = max(s_half_sum - s_disc, 0.0);

        // Reconstruct clamped Q = R * diag(s_lam) * R^T
        let s_atan_y = 2.0 * h_s12;
        let s_atan_x = h_s11 - h_s22;
        var s_theta = 0.0;
        if (abs(s_atan_y) > 1e-20 || abs(s_atan_x) > 1e-20) {
            s_theta = 0.5 * atan2(s_atan_y, s_atan_x);
        }
        let sc = cos(s_theta);
        let ss = sin(s_theta);
        let Q00 = sc * sc * s_lam1 + ss * ss * s_lam2;
        let Q01 = sc * ss * (s_lam1 - s_lam2);
        let Q11 = ss * ss * s_lam1 + sc * sc * s_lam2;

        // Twist/flip eigenvalues (PSD-clamped)
        let lam_twist = max(-k * Jm1 + mu, 0.0);
        let lam_flip = max(k * Jm1 + mu, 0.0);
        let a_coeff = 0.5 * (lam_twist + lam_flip);
        let b_coeff = 0.5 * (lam_flip - lam_twist);

        // Null-space coefficients
        let lam_n1 = select(0.0, max(p1 / sig1, 0.0), sig1 > 1e-8);
        let lam_n2 = select(0.0, max(p2 / sig2, 0.0), sig2 > 1e-8);

        // Weight vector for this vertex
        let wi = vec2f(dot(ci, v1), dot(ci, v2));

        // Compute diagonal Hessian block: H_ii = A0 * (5 rank-1 terms)
        let scale = A0;  // no dt² for VBD

        let wA1 = wi.x * wi.x;
        let wA2 = wi.x * wi.y;
        let wA3 = wi.y * wi.x;  // = wA2 for diagonal (wi == wj)
        let wA4 = wi.y * wi.y;

        let cc11 = Q00 * wA1 + a_coeff * wA4;
        let cc12 = Q01 * wA2 + b_coeff * wA3;
        let cc21 = Q01 * wA3 + b_coeff * wA2;
        let cc22 = Q11 * wA4 + a_coeff * wA1;
        let cc33 = lam_n1 * wA1 + lam_n2 * wA4;

        let r1 = cc11 * u1 + cc12 * u2;
        let r2 = cc21 * u1 + cc22 * u2;
        let r3 = cc33 * u3;

        let bx = scale * (u1.x * r1 + u2.x * r2 + u3.x * r3);
        let by = scale * (u1.y * r1 + u2.y * r2 + u3.y * r3);
        let bz = scale * (u1.z * r1 + u2.z * r2 + u3.z * r3);

        h00 += bx.x; h01 += bx.y; h02 += bx.z;
        h10 += by.x; h11 += by.y; h12 += by.z;
        h20 += bz.x; h21 += bz.y; h22 += bz.z;
    }

    // Accumulate to shared buffers
    gradient[vi] = gradient[vi] + vec4f(gx, gy, gz, 0.0);
    hessian[vi * 3u + 0u] = hessian[vi * 3u + 0u] + vec4f(h00, h01, h02, 0.0);
    hessian[vi * 3u + 1u] = hessian[vi * 3u + 1u] + vec4f(h10, h11, h12, 0.0);
    hessian[vi * 3u + 2u] = hessian[vi * 3u + 2u] + vec4f(h20, h21, h22, 0.0);
}
