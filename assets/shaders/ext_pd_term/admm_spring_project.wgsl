// ADMM Spring Local Projection
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b):
//   d = q[a] - q[b] + u[e]
//   z[e] = rest_length * normalize(d)

#import "core_simulate/header/solver_params.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> z: array<vec4f>;
@group(0) @binding(4) var<storage, read> u: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= solver.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;
    let rest_len = edge.rest_length;

    // d = S*q + u = (q[a] - q[b]) + u[e]
    let d = q[a].xyz - q[b].xyz + u[eid].xyz;
    let dist = length(d);

    if (dist > 1e-8) {
        z[eid] = vec4f(rest_len * d / dist, 0.0);
    } else {
        z[eid] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
