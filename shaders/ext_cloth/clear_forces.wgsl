// Clear force accumulation buffer (atomic u32 for CAS-based float atomics)
// Dispatch: ceil(node_count / 64) workgroups

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
@group(0) @binding(1) var<storage, read_write> forces: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let base = id * 4u;
    atomicStore(&forces[base + 0u], 0u);
    atomicStore(&forces[base + 1u], 0u);
    atomicStore(&forces[base + 2u], 0u);
    atomicStore(&forces[base + 3u], 0u);
}
