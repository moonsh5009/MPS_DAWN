// CG dot product — level 2 final reduction
// Sums all partial workgroup results into a single scalar.
// Dispatch: 1 workgroup

struct DotConfig {
    target_slot: u32,
    partial_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(2) var<uniform> config: DotConfig;

var<workgroup> shared_data: array<f32, 64>;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(local_invocation_id) lid: vec3u,
) {
    let local_id = lid.x;
    let count = config.partial_count;

    var sum = 0.0;
    var i = local_id;
    loop {
        if (i >= count) {
            break;
        }
        sum = sum + partials[i];
        i = i + 64u;
    }

    shared_data[local_id] = sum;
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            shared_data[local_id] = shared_data[local_id] + shared_data[local_id + stride];
        }
        workgroupBarrier();
    }

    if (local_id == 0u) {
        scalars[config.target_slot] = shared_data[0];
    }
}
