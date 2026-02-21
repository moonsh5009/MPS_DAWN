// CG initialization: x = 0, p = r (r already contains b from RHS assembly)
// Dispatch: ceil(node_count / 64) workgroups

#import "core_simulate/header/solver_params.wgsl"

@group(0) @binding(1) var<storage, read_write> cg_x: array<vec4f>;
@group(0) @binding(2) var<storage, read> cg_r: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cg_p: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let b = cg_r[id];
    cg_x[id] = vec4f(0.0, 0.0, 0.0, 0.0);
    cg_p[id] = b;
}
