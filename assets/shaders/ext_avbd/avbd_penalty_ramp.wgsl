// Augmented Lagrangian penalty ramp: penalty += beta * |C|, capped at stiffness.
// C = ||q[n0] - q[n1]|| - rest_length (spring constraint violation).
// Dispatch: ceil(edge_count / 64) after each VBD iteration.

#import "ext_avbd/header/al_params.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> al: ALParams;
@group(0) @binding(1) var<storage, read> q: array<vec4f>;
@group(0) @binding(2) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(3) var<storage, read_write> penalty: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= al.edge_count) {
        return;
    }

    let edge = edges[idx];
    let dx = q[edge.n0].xyz - q[edge.n1].xyz;
    let dist = length(dx);
    let C = dist - edge.rest_length;

    // Ramp penalty by beta * |C|, cap at material stiffness (Eq. 16)
    penalty[idx] = min(penalty[idx] + al.beta * abs(C), al.stiffness);
}
