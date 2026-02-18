// Normalize per-vertex normals from fixed-point i32 to unit vec4f
// Dispatch: ceil(node_count / 64) workgroups
//
// Reads accumulated area-weighted normals (i32 fixed-point),
// converts to float, normalizes, and writes to the output buffer
// used by the rendering vertex shader.

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

const FP_SCALE: f32 = 1048576.0; // 2^20

@group(0) @binding(0) var<uniform> params: ClothSimParams;
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
    var n = vec3f(0.0, 1.0, 0.0); // fallback normal (up)
    if (len > 1e-8) {
        n = raw / len;
    }

    normals_out[id] = vec4f(n, 0.0);
}
