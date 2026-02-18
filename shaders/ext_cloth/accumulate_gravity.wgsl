// Accumulate gravity forces using CAS-based atomic float add
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
@group(0) @binding(2) var<storage, read> mass: array<vec4f>;

fn atomicAddFloat(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old_val = atomicLoad(addr);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old_val) + val);
        let result = atomicCompareExchangeWeak(addr, old_val, new_val);
        if result.exchanged {
            break;
        }
        old_val = result.old_value;
    }
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let m = mass[id];
    let node_mass = m.x;
    let inv_mass = m.y;

    // Only apply gravity to non-pinned nodes (inv_mass > 0)
    if (inv_mass > 0.0) {
        let gravity = vec3f(params.gravity_x, params.gravity_y, params.gravity_z);
        let f = gravity * node_mass;

        let base = id * 4u;
        atomicAddFloat(&forces[base + 0u], f.x);
        atomicAddFloat(&forces[base + 1u], f.y);
        atomicAddFloat(&forces[base + 2u], f.z);
    }
}
