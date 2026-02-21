// Clear normal accumulation buffer (fixed-point i32 atomics)
// Dispatch: ceil(node_count / 64) workgroups

#import "ext_mesh/header/normal_params.wgsl"

@group(0) @binding(1) var<storage, read_write> normals: array<atomic<i32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let base = id * 4u;
    atomicStore(&normals[base + 0u], 0);
    atomicStore(&normals[base + 1u], 0);
    atomicStore(&normals[base + 2u], 0);
    atomicStore(&normals[base + 3u], 0);
}
