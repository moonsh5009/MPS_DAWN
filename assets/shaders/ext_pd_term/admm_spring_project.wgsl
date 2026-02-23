// ADMM Spring Local Projection (Proximal Operator)
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each edge (a,b) with physical stiffness k and ADMM penalty ρ:
//   v = q[a] - q[b] + u[e]
//   z[e] = [(k * rest_len + ρ * ||v||) / (k + ρ)] * normalize(v)
//
// When k >> ρ: z ≈ rest_len * normalize(v)  (hard projection)
// When ρ >> k: z ≈ v  (no projection)

#import "core_simulate/header/solver_params.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

struct ADMMSpringParams {
    penalty_weight: f32,  // ρ (ADMM penalty)
    stiffness: f32,       // k (physical stiffness)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> z: array<vec4f>;
@group(0) @binding(4) var<storage, read> u: array<vec4f>;
@group(0) @binding(5) var<uniform> admm_params: ADMMSpringParams;

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

    let k = admm_params.stiffness;
    let rho = admm_params.penalty_weight;

    // v = S*q + u = (q[a] - q[b]) + u[e]
    let v = q[a].xyz - q[b].xyz + u[eid].xyz;
    let v_len = length(v);

    if (v_len > 1e-8) {
        // Proximal operator: z_len = (k*L + ρ*||v||) / (k + ρ)
        let z_len = (k * rest_len + rho * v_len) / (k + rho);
        z[eid] = vec4f(z_len * v / v_len, 0.0);
    } else {
        z[eid] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
