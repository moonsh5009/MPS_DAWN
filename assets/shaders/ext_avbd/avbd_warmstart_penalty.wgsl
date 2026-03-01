// Augmented Lagrangian warmstart: penalty *= gamma, clamped to [1.0, stiffness].
// Dispatch: ceil(edge_count / 64) at frame start.

#import "ext_avbd/header/al_params.wgsl"

@group(0) @binding(0) var<uniform> al: ALParams;
@group(0) @binding(1) var<storage, read_write> penalty: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= al.edge_count) {
        return;
    }
    // Decay and clamp to [PENALTY_MIN, stiffness]
    penalty[idx] = clamp(penalty[idx] * al.gamma, 1.0, al.stiffness);
}
