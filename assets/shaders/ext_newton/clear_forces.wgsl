// Clear force accumulation buffer (atomic u32 for CAS-based float atomics)
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read_write> forces: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= solver.node_count) {
        return;
    }

    let base = id * 4u;
    atomicStore(&forces[base + 0u], 0u);
    atomicStore(&forces[base + 1u], 0u);
    atomicStore(&forces[base + 2u], 0u);
    atomicStore(&forces[base + 3u], 0u);
}
