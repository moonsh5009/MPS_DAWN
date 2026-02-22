// ADMM Area Reset for New Timestep
// Dispatch: ceil(face_count / 64) workgroups
//
// For each triangle (n0,n1,n2):
//   Compute F = Ds * Dm_inv from predicted positions s
//   z[f*2] = F col0,  z[f*2+1] = F col1   (warm-start)
//   u[f*2] = 0,  u[f*2+1] = 0              (reset dual)

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
@group(0) @binding(2) var<storage, read> s: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> z: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> u: array<vec4f>;

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

    // Predicted positions
    let x0 = s[na].xyz;
    let x1 = s[nb].xyz;
    let x2 = s[nc].xyz;

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

    // Warm-start z from deformation gradient of predicted positions
    z[fid * 2u] = vec4f(f0, 0.0);
    z[fid * 2u + 1u] = vec4f(f1, 0.0);

    // Reset dual variables
    u[fid * 2u] = vec4f(0.0, 0.0, 0.0, 0.0);
    u[fid * 2u + 1u] = vec4f(0.0, 0.0, 0.0, 0.0);
}
