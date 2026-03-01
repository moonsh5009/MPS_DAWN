// Blelloch exclusive prefix sum — local workgroup level.
// Each workgroup scans 256 elements and writes block sum to block_sums.
// Dispatch: ceil(scan_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> shared_data: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_main(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let local_id = lid.x;
    let global_id = wid.x * 256u + local_id;

    // Load into shared memory
    if (global_id < params.scan_count) {
        shared_data[local_id] = data[global_id];
    } else {
        shared_data[local_id] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduce)
    for (var stride = 1u; stride < 256u; stride = stride * 2u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            shared_data[idx] = shared_data[idx] + shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    // Save block total and set last element to 0
    if (local_id == 0u) {
        block_sums[wid.x] = shared_data[255u];
        shared_data[255u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    for (var stride = 128u; stride >= 1u; stride = stride >> 1u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            let temp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] = shared_data[idx] + temp;
        }
        workgroupBarrier();
    }

    // Write back
    if (global_id < params.scan_count) {
        data[global_id] = shared_data[local_id];
    }
}
