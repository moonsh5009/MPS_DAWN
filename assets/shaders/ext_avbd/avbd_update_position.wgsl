// AVBD update position: copy final positions from working iterate q.
// Pinned nodes (inv_mass <= 0) are restored to x_old.
// Free nodes get positions from q.
// Dispatch: ceil(node_count / 64) workgroups.
//
// Bindings:
//   0 = SolverParams (uniform)
//   1 = positions (storage, read_write)
//   2 = q (storage, read)
//   3 = x_old (storage, read)
//   4 = mass (storage, read)

#import "core_simulate/header/solver_params.wgsl"

struct SimMass {
    mass: f32,
    inv_mass: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> q: array<vec4f>;
@group(0) @binding(3) var<storage, read> x_old: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let inv_mass = mass[id].inv_mass;

    if (inv_mass > 0.0) {
        // Free node: use solved position from working iterate
        positions[id] = q[id];
    } else {
        // Pinned node: restore original position
        positions[id] = x_old[id];
    }
}
