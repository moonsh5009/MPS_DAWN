// ADMM Area Local Projection (ARAP rotation)
// Dispatch: ceil(face_count / 64) workgroups
//
// For each triangle (n0,n1,n2):
//   1. Compute deformation gradient F = Ds * Dm_inv (3x2)
//   2. G = F + u_reshaped (augmented Lagrangian)
//   3. SVD project G to closest rotation R = U * V^T
//   4. z[f*2] = R col0,  z[f*2+1] = R col1

#import "core_simulate/header/solver_params.wgsl"

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

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> triangles: array<AreaTriangle>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> z: array<vec4f>;
@group(0) @binding(4) var<storage, read> u: array<vec4f>;

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

    // G = F + u (augmented Lagrangian term)
    let g0 = f0 + u[fid * 2u].xyz;
    let g1 = f1 + u[fid * 2u + 1u].xyz;

    // Cauchy-Green C = G^T * G (2x2 symmetric)
    let c00 = dot(g0, g0);
    let c01 = dot(g0, g1);
    let c11 = dot(g1, g1);
    let det_C = c00 * c11 - c01 * c01;

    // Default: identity rotation
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
        let u1 = (g0 * v1.x + g1 * v1.y) / sig1;
        let u2 = (g0 * v2.x + g1 * v2.y) / sig2;

        // Closest rotation R = U * V^T
        r0 = u1 * v1.x + u2 * v2.x;
        r1 = u1 * v1.y + u2 * v2.y;
    }

    z[fid * 2u] = vec4f(r0, 0.0);
    z[fid * 2u + 1u] = vec4f(r1, 0.0);
}
