// Update velocity: v = (v + dv_total) * damping (for free nodes)
// Dispatch: ceil(node_count / 64) workgroups
//
// dv_total contains the accumulated Newton velocity delta.
// Pinned nodes (inv_mass == 0) have velocity set to zero.

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
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(2) var<storage, read> dv_total: array<vec4f>;
@group(0) @binding(3) var<storage, read> mass: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let inv_mass = mass[id].y;

    if (inv_mass > 0.0) {
        // Free node: apply accumulated velocity delta and damping
        let v = velocities[id].xyz;
        let dv = dv_total[id].xyz;
        let new_v = (v + dv) * params.damping;
        velocities[id] = vec4f(new_v, 0.0);
    } else {
        // Pinned node: zero velocity
        velocities[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
