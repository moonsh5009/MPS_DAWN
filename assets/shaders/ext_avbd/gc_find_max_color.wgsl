// Reduction: find maximum color value across all vertices.
// Uses shared memory reduction + atomicMax on global output.
// Dispatch: ceil(node_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read> colors: array<u32>;
@group(0) @binding(2) var<storage, read_write> max_color: array<atomic<u32>>;

var<workgroup> shared_max: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let global_id = gid.x;
    let local_id = lid.x;

    if (global_id < params.node_count) {
        shared_max[local_id] = colors[global_id];
    } else {
        shared_max[local_id] = 0u;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
        }
        workgroupBarrier();
    }

    if (local_id == 0u) {
        atomicMax(&max_color[0], shared_max[0]);
    }
}
