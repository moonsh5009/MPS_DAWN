// ADMM Spring Dual Variable Update
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b):
//   u[e] += S*q - z[e]
//   where S*q = q[a] - q[b]

#import "core_simulate/header/solver_params.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read> z: array<vec4f>;
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

    // S*q = q[a] - q[b]
    let sq = q[a].xyz - q[b].xyz;

    // u += S*q - z
    u[eid] = vec4f(u[eid].xyz + sq - z[eid].xyz, 0.0);
}
