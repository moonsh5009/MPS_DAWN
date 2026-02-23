// Jacobi preconditioner: z = D^{-1} * r
// D stored as 3x3 blocks (9 floats per node); uses diagonal elements only.
// Also applies MPCG filter (z = 0 for pinned nodes).
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> cg_r: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_z: array<vec4f>;
@group(0) @binding(3) var<storage, read> diag: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;
    if (inv_mass <= 0.0) {
        cg_z[id] = vec4f(0.0);
        return;
    }

    let r = cg_r[id].xyz;
    let d00 = diag[id * 9u + 0u];
    let d11 = diag[id * 9u + 4u];
    let d22 = diag[id * 9u + 8u];

    let z = vec3f(
        select(0.0, r.x / d00, abs(d00) > 1e-20),
        select(0.0, r.y / d11, abs(d11) > 1e-20),
        select(0.0, r.z / d22, abs(d22) > 1e-20),
    );
    cg_z[id] = vec4f(z, 0.0);
}
