// CG dot product — level 1 workgroup reduction
// Computes partial sums of dot(a[i].xyz, b[i].xyz) per workgroup
// Dispatch: ceil(node_count / 64) workgroups
//
// Each workgroup reduces its local elements and writes one partial sum.
// A second pass (cg_dot_final) reduces the partials to a single scalar.

struct ClothSimParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    node_count: u32,
    edge_count: u32,
    face_count: u32,
    cg_max_iter: u32,
    damping: f32,
    cg_tolerance: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> params: ClothSimParams;
@group(0) @binding(1) var<storage, read> vec_a: array<vec4f>;
@group(0) @binding(2) var<storage, read> vec_b: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> shared_data: array<f32, 64>;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let global_id = gid.x;
    let local_id = lid.x;

    // Load and compute local dot product
    var val = 0.0;
    if (global_id < params.node_count) {
        let a = vec_a[global_id].xyz;
        let b = vec_b[global_id].xyz;
        val = dot(a, b);
    }

    shared_data[local_id] = val;
    workgroupBarrier();

    // Parallel reduction within workgroup
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            shared_data[local_id] = shared_data[local_id] + shared_data[local_id + stride];
        }
        workgroupBarrier();
    }

    // Write workgroup result
    if (local_id == 0u) {
        partials[wid.x] = shared_data[0];
    }
}
