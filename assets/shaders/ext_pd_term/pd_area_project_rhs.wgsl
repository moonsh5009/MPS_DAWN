// PD Area Fused: ARAP Local Projection + RHS Assembly
// Dispatch: ceil(face_count / 64) workgroups
//
// For each triangle (n0,n1,n2) with weight w = stiffness * rest_area:
//   1. Compute deformation gradient F = Ds * Dm_inv (3x2)
//   2. SVD projection: F -> R (closest rotation via U*V^T)
//   3. Scatter to RHS: rhs[n_alpha] += w * S^T * R
// Eliminates the intermediate projection buffer.

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

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> triangles: array<AreaTriangle>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rhs: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> area_params: AreaParams;

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

    // Current positions
    let x0 = q[na].xyz;
    let x1 = q[nb].xyz;
    let x2 = q[nc].xyz;

    // Deformed edges (3D)
    let ds0 = x1 - x0;
    let ds1 = x2 - x0;

    // Deformation gradient F = Ds * Dm_inv (3x2)
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

    // Degenerate triangle: use identity rotation
    var r0 = vec3f(1.0, 0.0, 0.0);
    var r1 = vec3f(0.0, 1.0, 0.0);

    if (det_C >= 1e-20) {
        // SVD via eigendecomposition of C
        let half_sum = 0.5 * (c00 + c11);
        let half_diff = 0.5 * (c00 - c11);
        let disc = sqrt(half_diff * half_diff + c01 * c01);
        let lam1 = max(half_sum + disc, 1e-12);
        let lam2 = max(half_sum - disc, 1e-12);

        let sig1 = sqrt(lam1);
        let sig2 = sqrt(lam2);

        // Eigenvectors of C -> V rotation matrix
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

        // Left singular vectors
        let u1 = (f0 * v1.x + f1 * v1.y) / sig1;
        let u2 = (f0 * v2.x + f1 * v2.y) / sig2;

        // Closest rotation R = U * V^T
        r0 = u1 * v1.x + u2 * v2.x;
        r1 = u1 * v1.y + u2 * v2.y;
    }

    // S^T distribution via Dm_inv coefficients
    let ci1 = vec2f(tri.dm_inv_00, tri.dm_inv_01);  // vertex 1 (n1)
    let ci2 = vec2f(tri.dm_inv_10, tri.dm_inv_11);  // vertex 2 (n2)
    let ci0 = -(ci1 + ci2);                          // vertex 0 (n0)

    let contrib0 = w * (ci0.x * r0 + ci0.y * r1);
    let contrib1 = w * (ci1.x * r0 + ci1.y * r1);
    let contrib2 = w * (ci2.x * r0 + ci2.y * r1);

    // Scatter to RHS
    let base_a = na * 4u;
    atomicAddFloat(&rhs[base_a + 0u], contrib0.x);
    atomicAddFloat(&rhs[base_a + 1u], contrib0.y);
    atomicAddFloat(&rhs[base_a + 2u], contrib0.z);

    let base_b = nb * 4u;
    atomicAddFloat(&rhs[base_b + 0u], contrib1.x);
    atomicAddFloat(&rhs[base_b + 1u], contrib1.y);
    atomicAddFloat(&rhs[base_b + 2u], contrib1.z);

    let base_c = nc * 4u;
    atomicAddFloat(&rhs[base_c + 0u], contrib2.x);
    atomicAddFloat(&rhs[base_c + 1u], contrib2.y);
    atomicAddFloat(&rhs[base_c + 2u], contrib2.z);
}
