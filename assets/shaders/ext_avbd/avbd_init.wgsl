// AVBD init: save current positions into x_old buffer.
// Dispatch: ceil(node_count / 64) workgroups.
//
// Bindings:
//   0 = SolverParams (uniform)
//   1 = positions (storage, read)
//   2 = x_old (storage, read_write)

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> x_old: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    x_old[id] = positions[id];
}
