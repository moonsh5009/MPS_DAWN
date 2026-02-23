// JGS2 inertia accumulation: initialize gradient and Hessian diagonal per vertex
// Dispatch: ceil(node_count / 64) workgroups
//
// Must run BEFORE spring term accumulation (springs use atomicAddFloat on top).
// Uses atomicStore (safe — each thread writes a unique vertex).
//
// Gradient: g_i = M_i/dt² * (q_i - s_i)
// Hessian:  H_ii = M_i/dt² * I₃

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read> s: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;
@group(0) @binding(5) var<storage, read_write> gradient: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> hessian_diag: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;
    let base_g = id * 4u;
    let base_h = id * 9u;

    if (inv_mass <= 0.0) {
        // Pinned node: zero gradient and Hessian
        for (var j = 0u; j < 4u; j = j + 1u) {
            atomicStore(&gradient[base_g + j], 0u);
        }
        for (var j = 0u; j < 9u; j = j + 1u) {
            atomicStore(&hessian_diag[base_h + j], 0u);
        }
        return;
    }

    let m = mass[id].mass;
    let M_dt2 = m * physics.inv_dt_sq;

    // Gradient: g = M/dt² * (q - s)
    let qi = q[id].xyz;
    let si = s[id].xyz;
    let g = M_dt2 * (qi - si);

    atomicStore(&gradient[base_g + 0u], bitcast<u32>(g.x));
    atomicStore(&gradient[base_g + 1u], bitcast<u32>(g.y));
    atomicStore(&gradient[base_g + 2u], bitcast<u32>(g.z));
    atomicStore(&gradient[base_g + 3u], 0u);

    // Hessian diagonal: H = M/dt² * I₃ (row-major 3x3)
    let h_val = bitcast<u32>(M_dt2);
    atomicStore(&hessian_diag[base_h + 0u], h_val);  // [0,0]
    atomicStore(&hessian_diag[base_h + 1u], 0u);      // [0,1]
    atomicStore(&hessian_diag[base_h + 2u], 0u);      // [0,2]
    atomicStore(&hessian_diag[base_h + 3u], 0u);      // [1,0]
    atomicStore(&hessian_diag[base_h + 4u], h_val);  // [1,1]
    atomicStore(&hessian_diag[base_h + 5u], 0u);      // [1,2]
    atomicStore(&hessian_diag[base_h + 6u], 0u);      // [2,0]
    atomicStore(&hessian_diag[base_h + 7u], 0u);      // [2,1]
    atomicStore(&hessian_diag[base_h + 8u], h_val);  // [2,2]
}
