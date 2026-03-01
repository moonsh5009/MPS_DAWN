// Propagate scanned block offsets back to elements.
// Each element: data[i] += block_offsets[i / 256]
// Dispatch: ceil(scan_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> block_offsets: array<u32>;

@compute @workgroup_size(256)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let global_id = gid.x;
    if (global_id >= params.scan_count) {
        return;
    }

    data[global_id] = data[global_id] + block_offsets[wid.x];
}
