// Update position: pos = x_old + vel * dt (for free nodes)
// Dispatch: ceil(node_count / 64) workgroups
//
// Uses x_old (saved at Newton init) as base position.
// Velocity already includes dv_total and damping from update_velocity.
// Pinned nodes (inv_mass == 0) are not moved.

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
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> x_old: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let inv_mass = mass[id].y;

    if (inv_mass > 0.0) {
        positions[id] = vec4f(x_old[id].xyz + velocities[id].xyz * params.dt, 1.0);
    }
}
