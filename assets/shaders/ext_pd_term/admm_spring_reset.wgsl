// ADMM Spring Reset for New Timestep
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b):
//   z[e] = S*s = s[a] - s[b]   (warm-start from predicted positions)
//   u[e] = 0                    (reset dual variable)

#import "core_simulate/header/solver_params.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read> s: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> z: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> u: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= solver.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;

    // Warm-start z from predicted positions
    z[eid] = vec4f(s[a].xyz - s[b].xyz, 0.0);

    // Reset dual variable
    u[eid] = vec4f(0.0, 0.0, 0.0, 0.0);
}
