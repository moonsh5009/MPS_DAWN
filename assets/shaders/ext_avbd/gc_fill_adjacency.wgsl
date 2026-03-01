// Fill CSR col_idx from edge list using atomic write counters.
// Per edge: write both directions (n0→n1 and n1→n0) into col_idx.
// Dispatch: ceil(edge_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

struct MeshEdge {
    n0: u32,
    n1: u32,
};

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read> edges: array<MeshEdge>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read_write> write_counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> col_idx: array<u32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let edge_id = gid.x;
    if (edge_id >= params.edge_count) {
        return;
    }

    let e = edges[edge_id];

    // n0 → n1
    let offset0 = row_ptr[e.n0] + atomicAdd(&write_counters[e.n0], 1u);
    col_idx[offset0] = e.n1;

    // n1 → n0
    let offset1 = row_ptr[e.n1] + atomicAdd(&write_counters[e.n1], 1u);
    col_idx[offset1] = e.n0;
}
