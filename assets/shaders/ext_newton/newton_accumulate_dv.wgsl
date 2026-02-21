// Newton accumulate dv: dv_total += cg_x (CG solution for this Newton step)
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(1) var<storage, read_write> dv_total: array<vec4f>;
@group(0) @binding(2) var<storage, read> cg_x: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    dv_total[id] = dv_total[id] + cg_x[id];
}
