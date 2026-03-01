// Greedy first-fit vertex coloring (1 iteration per dispatch).
// Per vertex: if uncolored and all lower-indexed neighbors are colored,
// find smallest unused color via bitmask.
// Dispatch: ceil(node_count / 256) workgroups.

#import "ext_avbd/header/coloring_params.wgsl"

const UNCOLORED: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> params: ColoringParams;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> colors: array<u32>;
@group(0) @binding(4) var<storage, read_write> flag: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let v = gid.x;
    if (v >= params.node_count) {
        return;
    }

    // Skip already colored
    if (colors[v] != UNCOLORED) {
        return;
    }

    let row_start = row_ptr[v];
    let row_end = row_ptr[v + 1u];

    // Check if any lower-indexed neighbor is still uncolored
    for (var i = row_start; i < row_end; i = i + 1u) {
        let neighbor = col_idx[i];
        if (neighbor < v && colors[neighbor] == UNCOLORED) {
            // Must wait — set flag to indicate not done
            atomicMax(&flag[0], 1u);
            return;
        }
    }

    // Collect used colors into 64-bit bitmask (2 × u32)
    var used_lo: u32 = 0u;  // colors 0..31
    var used_hi: u32 = 0u;  // colors 32..63

    for (var i = row_start; i < row_end; i = i + 1u) {
        let c = colors[col_idx[i]];
        if (c != UNCOLORED) {
            if (c < 32u) {
                used_lo = used_lo | (1u << c);
            } else if (c < 64u) {
                used_hi = used_hi | (1u << (c - 32u));
            }
        }
    }

    // Find first unused color via firstTrailingBit on inverted bitmask
    let free_lo = ~used_lo;
    if (free_lo != 0u) {
        colors[v] = firstTrailingBit(free_lo);
    } else {
        let free_hi = ~used_hi;
        if (free_hi != 0u) {
            colors[v] = 32u + firstTrailingBit(free_hi);
        } else {
            // Fallback: more than 64 colors needed (shouldn't happen for reasonable meshes)
            colors[v] = 64u;
        }
    }
}
