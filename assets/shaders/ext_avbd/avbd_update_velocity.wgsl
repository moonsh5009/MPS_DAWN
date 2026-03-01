// AVBD update velocity: v = (q - x_old) * inv_dt * damping (for free nodes).
// Pinned nodes (inv_mass <= 0) get zero velocity.
// Dispatch: ceil(node_count / 64) workgroups.
//
// Bindings:
//   0 = PhysicsParams (uniform)
//   1 = SolverParams (uniform)
//   2 = velocities (storage, read_write)
//   3 = q (storage, read)
//   4 = x_old (storage, read)
//   5 = mass (storage, read)

#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/solver_params.wgsl"

struct SimMass {
    mass: f32,
    inv_mass: f32,
};

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> q: array<vec4f>;
@group(0) @binding(4) var<storage, read> x_old: array<vec4f>;
@group(0) @binding(5) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        // Free node: v = (q - x_old) / dt * damping
        let displacement = q[id].xyz - x_old[id].xyz;
        velocities[id] = vec4f(displacement * physics.inv_dt * physics.damping, 0.0);
    } else {
        // Pinned node: zero velocity
        velocities[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
