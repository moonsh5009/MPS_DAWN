// Count vertex degrees from edge list.
// Per edge: atomicAdd degree[n0] and degree[n1].
// Dispatch: ceil(edge_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

struct MeshEdge {
    n0: u32,
    n1: u32,
};

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read> edges: array<MeshEdge>;
@group(0) @binding(2) var<storage, read_write> degrees: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let edge_id = gid.x;
    if (edge_id >= params.edge_count) {
        return;
    }

    let e = edges[edge_id];
    atomicAdd(&degrees[e.n0], 1u);
    atomicAdd(&degrees[e.n1], 1u);
}
