// Newton predict positions: x_temp = x_old + dt * (v + dv_total)
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/physics_params.wgsl"
#import "core_simulate/header/sim_mass.wgsl"

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> solver: SolverParams;

@group(0) @binding(2) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read> x_old: array<vec4f>;
@group(0) @binding(4) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(5) var<storage, read> dv_total: array<vec4f>;
@group(0) @binding(6) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        let vel = velocities[id].xyz;
        let dv = dv_total[id].xyz;
        positions[id] = vec4f(x_old[id].xyz + physics.dt * (vel + dv), 1.0);
    } else {
        // Pinned node stays at original position
        positions[id] = x_old[id];
    }
}
