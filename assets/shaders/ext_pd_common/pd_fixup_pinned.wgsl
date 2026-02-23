// Fixup pinned nodes: restore q from s for nodes with inv_mass == 0.
// ADMM CG solves for absolute positions starting from x=0, but MPCG
// zeros pinned node variables, leaving them at origin. This shader
// restores pinned nodes to their predicted positions (s = x_old).
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> q: array<vec4f>;
@group(0) @binding(2) var<storage, read> s: array<vec4f>;
@group(0) @binding(3) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;
    if (inv_mass <= 0.0) {
        q[id] = s[id];
    }
}
