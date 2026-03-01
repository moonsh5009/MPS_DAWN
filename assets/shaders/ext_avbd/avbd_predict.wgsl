// AVBD predict: compute predicted position s = x_old + dt*v + dt^2*gravity.
// Pinned nodes (inv_mass <= 0) get s = x_old.
// Dispatch: ceil(node_count / 64) workgroups.
//
// Bindings:
//   0 = PhysicsParams (uniform)
//   1 = SolverParams (uniform)
//   2 = x_old (storage, read)
//   3 = velocities (storage, read)
//   4 = mass (storage, read)
//   5 = s (storage, read_write)

#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/solver_params.wgsl"

struct SimMass {
    mass: f32,
    inv_mass: f32,
};

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;
@group(0) @binding(2) var<storage, read> x_old: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;
@group(0) @binding(5) var<storage, read_write> s: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        // Free node: s = x_old + dt*v + dt^2*gravity
        let x = x_old[id].xyz;
        let v = velocities[id].xyz;
        let gravity = vec3f(physics.gravity_x, physics.gravity_y, physics.gravity_z);
        s[id] = vec4f(x + physics.dt * v + physics.dt_sq * gravity, 1.0);
    } else {
        // Pinned node: s = x_old
        s[id] = x_old[id];
    }
}
