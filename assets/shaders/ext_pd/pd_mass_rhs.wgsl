// Add inertial RHS: rhs += (M / dt^2) * s
// Uses CAS-based atomic float addition for concurrent accumulation.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

@group(0) @binding(2) var<storage, read> mass: array<SimMass>;
@group(0) @binding(3) var<storage, read> s: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rhs: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let m = mass[id].mass;
    let coeff = m * physics.inv_dt_sq;
    let sv = s[id].xyz;

    let base = id * 4u;
    atomicAddFloat(&rhs[base + 0u], coeff * sv.x);
    atomicAddFloat(&rhs[base + 1u], coeff * sv.y);
    atomicAddFloat(&rhs[base + 2u], coeff * sv.z);
}
