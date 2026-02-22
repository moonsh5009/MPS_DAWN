// ADMM Spring RHS Assembly
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b) with weight w = stiffness:
//   val = w * (z[e] - u[e])
//   rhs[a] += val,  rhs[b] -= val
// Distributes S^T * (z - u) weighted by stiffness.

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
@group(0) @binding(2) var<storage, read> z: array<vec4f>;
@group(0) @binding(3) var<storage, read> u: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rhs: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> spring_params: SpringParams;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= solver.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;
    let w = spring_params.stiffness;

    // val = w * (z[e] - u[e])
    let val = w * (z[eid].xyz - u[eid].xyz);

    // Scatter S^T: rhs[a] += val, rhs[b] -= val
    let base_a = a * 4u;
    atomicAddFloat(&rhs[base_a + 0u], val.x);
    atomicAddFloat(&rhs[base_a + 1u], val.y);
    atomicAddFloat(&rhs[base_a + 2u], val.z);

    let base_b = b * 4u;
    atomicAddFloat(&rhs[base_b + 0u], -val.x);
    atomicAddFloat(&rhs[base_b + 1u], -val.y);
    atomicAddFloat(&rhs[base_b + 2u], -val.z);
}
