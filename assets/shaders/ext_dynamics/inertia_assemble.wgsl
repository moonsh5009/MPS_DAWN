// Add mass (inertia) to A diagonal: diag[i] += M_i * I3x3
// where I3x3 is the 3x3 identity matrix.
// Dispatch: ceil(node_count / 64) workgroups
//
// The diagonal buffer stores 3x3 blocks (9 f32 per node, row-major).
// InertialTerm writes M_i to entries [0,0], [1,1], [2,2] only.
// Must run AFTER clear_hessian and BEFORE any atomic-based term (SpringTerm).

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(1) var<storage, read_write> diag_values: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let m = mass[id].mass;
    let base = id * 9u;

    // Add M to diagonal entries [0,0], [1,1], [2,2]
    diag_values[base + 0u] = m;  // [0,0]
    diag_values[base + 4u] = m;  // [1,1]
    diag_values[base + 8u] = m;  // [2,2]
}
