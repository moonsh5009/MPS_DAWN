// AVBD spring accumulation: adds spring gradient + PSD-projected Hessian per vertex.
// Vertex-centric: each thread loops over CSR neighbors (gather pattern).
// Augmented Lagrangian: uses per-edge penalty parameter (ramped toward stiffness k).
// Both gradient and Hessian use penalty (not k) for consistent Newton steps.
// Dispatch: ceil(color_vertex_count / 64) per color group.

#import "ext_avbd/header/vbd_params.wgsl"

struct SpringNeighbor {
    neighbor_idx: u32,
    rest_length: f32,
    edge_index: u32,
};

@group(0) @binding(0) var<uniform> color_params: VBDColorParams;
@group(0) @binding(1) var<storage, read> q: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> gradient: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> hessian: array<vec4f>;
@group(0) @binding(4) var<storage, read> vertex_order: array<u32>;
@group(0) @binding(5) var<storage, read> spring_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> spring_neighbors: array<SpringNeighbor>;
@group(0) @binding(7) var<storage, read> penalty: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let local_idx = gid.x;
    if (local_idx >= color_params.color_vertex_count) {
        return;
    }

    let vi = vertex_order[color_params.color_offset + local_idx];
    let qi = q[vi].xyz;
    let start = spring_offsets[vi];
    let end = spring_offsets[vi + 1u];

    var gx = 0.0;
    var gy = 0.0;
    var gz = 0.0;
    var h00 = 0.0; var h01 = 0.0; var h02 = 0.0;
    var h10 = 0.0; var h11 = 0.0; var h12 = 0.0;
    var h20 = 0.0; var h21 = 0.0; var h22 = 0.0;

    for (var e = start; e < end; e = e + 1u) {
        let neighbor = spring_neighbors[e];
        let j = neighbor.neighbor_idx;
        let rest_len = neighbor.rest_length;

        let dx = qi - q[j].xyz;
        let dist = length(dx);

        if (dist < 1e-8) {
            continue;
        }

        let dir = dx / dist;

        // Per-edge penalty parameter (ramped toward stiffness k)
        let p = penalty[neighbor.edge_index];

        // Energy gradient: penalty * C * dir (consistent with Hessian)
        let C = dist - rest_len;
        let g_spring = p * C;
        gx += g_spring * dir.x;
        gy += g_spring * dir.y;
        gz += g_spring * dir.z;

        // PSD-projected Hessian: penalty * [(1-ratio)*I + ratio*(dir⊗dir)]
        // Uses penalty (not k) for consistency with gradient
        let ratio = min(rest_len / dist, 1.0);
        let coeff_i = p * (1.0 - ratio);
        let coeff_d = p * ratio;

        h00 += coeff_i + coeff_d * dir.x * dir.x;
        h01 += coeff_d * dir.x * dir.y;
        h02 += coeff_d * dir.x * dir.z;
        h10 += coeff_d * dir.y * dir.x;
        h11 += coeff_i + coeff_d * dir.y * dir.y;
        h12 += coeff_d * dir.y * dir.z;
        h20 += coeff_d * dir.z * dir.x;
        h21 += coeff_d * dir.z * dir.y;
        h22 += coeff_i + coeff_d * dir.z * dir.z;
    }

    // Accumulate (add) to existing gradient/hessian from inertia
    gradient[vi] = gradient[vi] + vec4f(gx, gy, gz, 0.0);
    hessian[vi * 3u + 0u] = hessian[vi * 3u + 0u] + vec4f(h00, h01, h02, 0.0);
    hessian[vi * 3u + 1u] = hessian[vi * 3u + 1u] + vec4f(h10, h11, h12, 0.0);
    hessian[vi * 3u + 2u] = hessian[vi * 3u + 2u] + vec4f(h20, h21, h22, 0.0);
}
