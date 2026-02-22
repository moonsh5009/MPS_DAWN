// Accumulate gravity forces using CAS-based atomic float add
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let node_mass = mass[id].mass;
    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        let gravity = vec3f(physics.gravity_x, physics.gravity_y, physics.gravity_z);
        let f = gravity * node_mass;

        let base = id * 4u;
        atomicAddFloat(&forces[base + 0u], f.x);
        atomicAddFloat(&forces[base + 1u], f.y);
        atomicAddFloat(&forces[base + 2u], f.z);
    }
}
