// CG direction update: p = r + beta * p
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(1) var<storage, read> cg_r: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_p: array<vec4f>;
@group(0) @binding(3) var<storage, read> scalars: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let beta = scalars[4];

    let r = cg_r[id].xyz;
    let p = cg_p[id].xyz;

    cg_p[id] = vec4f(r + beta * p, 0.0);

    // MPCG filter: zero search direction for pinned (infinite mass) nodes
    let inv_mass = mass[id].inv_mass;
    if (inv_mass <= 0.0) {
        cg_p[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
