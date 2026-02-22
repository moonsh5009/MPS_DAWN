// PD Spring Fused: Local Projection + RHS Assembly
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b) with weight w = stiffness:
//   1. Compute projection: p = rest_length * normalize(q[a] - q[b])
//   2. Scatter to RHS:     rhs[a] += w * p,  rhs[b] -= w * p
// Eliminates the intermediate projection buffer.

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

struct SpringParams {
    stiffness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rhs: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> spring_params: SpringParams;

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
    let w = spring_params.stiffness;

    // Local projection: p = rest_length * normalize(q[a] - q[b])
    let qa = q[a].xyz;
    let qb = q[b].xyz;
    let dx = qa - qb;
    let dist = length(dx);

    if (dist <= 1e-8) {
        return;
    }

    let p = rest_len * dx / dist;
    let wp = w * p;

    // Scatter to RHS: rhs[a] += w*p, rhs[b] -= w*p
    let base_a = a * 4u;
    atomicAddFloat(&rhs[base_a + 0u], wp.x);
    atomicAddFloat(&rhs[base_a + 1u], wp.y);
    atomicAddFloat(&rhs[base_a + 2u], wp.z);

    let base_b = b * 4u;
    atomicAddFloat(&rhs[base_b + 0u], -wp.x);
    atomicAddFloat(&rhs[base_b + 1u], -wp.y);
    atomicAddFloat(&rhs[base_b + 2u], -wp.z);
}
