// CG update: x += alpha * p, r -= alpha * Ap
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> cg_x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_r: array<vec4f>;
@group(0) @binding(3) var<storage, read> cg_p: array<vec4f>;
@group(0) @binding(4) var<storage, read> cg_ap: array<vec4f>;
@group(0) @binding(5) var<storage, read> scalars: array<f32>;
@group(0) @binding(6) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let alpha = scalars[3];

    let x = cg_x[id].xyz;
    let r = cg_r[id].xyz;
    let p = cg_p[id].xyz;
    let ap = cg_ap[id].xyz;

    cg_x[id] = vec4f(x + alpha * p, 0.0);
    cg_r[id] = vec4f(r - alpha * ap, 0.0);

    // MPCG filter: zero residual for pinned (infinite mass) nodes
    let inv_mass = mass[id].inv_mass;
    if (inv_mass <= 0.0) {
        cg_r[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
