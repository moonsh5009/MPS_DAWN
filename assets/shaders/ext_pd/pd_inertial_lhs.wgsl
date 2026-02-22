// Add M/dt^2 to LHS diagonal (3x3 identity blocks)
// Uses CAS-based atomic float addition for concurrent accumulation.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

@group(0) @binding(2) var<storage, read> mass: array<SimMass>;
@group(0) @binding(3) var<storage, read_write> diag: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let m = mass[id].mass;
    let val = m * physics.inv_dt_sq;

    // Add val to diagonal entries of the 3x3 block: indices 0, 4, 8
    let base = id * 9u;
    atomicAddFloat(&diag[base + 0u], val);
    atomicAddFloat(&diag[base + 4u], val);
    atomicAddFloat(&diag[base + 8u], val);
}
