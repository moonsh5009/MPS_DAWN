// Update velocity: v = (v + dv_total) * damping (for free nodes)
// Dispatch: ceil(node_count / 64) workgroups
//
// dv_total contains the accumulated Newton velocity delta.
// Pinned nodes (inv_mass == 0) have velocity set to zero.

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> dv_total: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        // Free node: apply accumulated velocity delta and damping
        let v = velocities[id].xyz;
        let dv = dv_total[id].xyz;
        var new_v = (v + dv) * physics.damping;

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
