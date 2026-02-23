// CG initialization: x = 0, p = r (r already contains b from RHS assembly)
// MPCG: zero r and p for pinned nodes (inv_mass <= 0) to prevent huge
// pinned-node RHS values from dominating the initial CG direction.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> cg_x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_r: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cg_p: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    cg_x[id] = vec4f(0.0, 0.0, 0.0, 0.0);

    let inv_mass = mass[id].inv_mass;
    if (inv_mass <= 0.0) {
        // Pinned node: zero residual and direction
        cg_r[id] = vec4f(0.0, 0.0, 0.0, 0.0);
        cg_p[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    } else {
        cg_p[id] = cg_r[id];
    }
}
