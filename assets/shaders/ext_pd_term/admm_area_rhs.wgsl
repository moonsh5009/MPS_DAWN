// ADMM Area RHS Assembly
// Dispatch: ceil(face_count / 64) workgroups
//
// For each triangle (n0,n1,n2) with weight w = stiffness * rest_area:
//   diff0 = z[f*2] - u[f*2],  diff1 = z[f*2+1] - u[f*2+1]
//   Scatter S^T * (z - u) weighted by w to rhs[n0], rhs[n1], rhs[n2]
//   using Dm_inv coefficient vectors.

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
@group(0) @binding(2) var<storage, read> z: array<vec4f>;
@group(0) @binding(3) var<storage, read> u: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rhs: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> area_params: AreaParams;

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

    // z - u difference (3x2 stored as 2 vec4f per face)
    let diff0 = z[fid * 2u].xyz - u[fid * 2u].xyz;
    let diff1 = z[fid * 2u + 1u].xyz - u[fid * 2u + 1u].xyz;

    // S^T distribution via Dm_inv coefficients
    let ci1 = vec2f(tri.dm_inv_00, tri.dm_inv_01);  // vertex 1 (n1)
    let ci2 = vec2f(tri.dm_inv_10, tri.dm_inv_11);  // vertex 2 (n2)
    let ci0 = -(ci1 + ci2);                          // vertex 0 (n0)

    let contrib0 = w * (ci0.x * diff0 + ci0.y * diff1);
    let contrib1 = w * (ci1.x * diff0 + ci1.y * diff1);
    let contrib2 = w * (ci2.x * diff0 + ci2.y * diff1);

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
