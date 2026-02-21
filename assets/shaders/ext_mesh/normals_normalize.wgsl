// Normalize per-vertex normals from fixed-point i32 to unit vec4f
// Dispatch: ceil(node_count / 64) workgroups

#import "ext_mesh/header/normal_params.wgsl"

const FP_SCALE: f32 = 1048576.0; // 2^20

@group(0) @binding(1) var<storage, read> normals_i32: array<i32>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let base = id * 4u;
    let raw = vec3f(
        f32(normals_i32[base + 0u]) / FP_SCALE,
        f32(normals_i32[base + 1u]) / FP_SCALE,
        f32(normals_i32[base + 2u]) / FP_SCALE,
    );

    let len = length(raw);
    var n = vec3f(0.0, 1.0, 0.0);
    if (len > 1e-8) {
        n = raw / len;
    }

    normals_out[id] = vec4f(n, 0.0);
}
