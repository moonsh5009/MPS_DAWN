// Update velocity from PD result: v = (q - x_old) / dt * damping
// Pinned nodes (inv_mass == 0) get zero velocity.
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

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
        var new_v = (q[id].xyz - x_old[id].xyz) * physics.inv_dt * physics.damping;

        // Safety clamp: prevent velocity explosion
        let speed = length(new_v);
        let max_speed = 50.0;  // m/s — generous upper bound
        if (speed > max_speed) {
            new_v = new_v * (max_speed / speed);
        }

        velocities[id] = vec4f(new_v, 0.0);
    } else {
        // Pinned node: zero velocity
        velocities[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
