// CG sparse matrix-vector product: Ap = A * p
// where A = M - dt^2 * H (system matrix)
// Dispatch: ceil(node_count / 64) workgroups
//
// Per node i:
//   Ap[i] = M[i] * p[i] - dt^2 * (H_diag[i] * p[i] + sum_j H_offdiag[i,j] * p[j])
//
// M[i] = mass[i].x (scalar mass, applied as scalar * vec3)
// H off-diagonal stored in CSR format (3x3 blocks)
// H diagonal stored as f32 (3x3 blocks, node_count * 9)

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
@group(0) @binding(1) var<storage, read> cg_p: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cg_ap: array<vec4f>;
@group(0) @binding(3) var<storage, read> mass: array<vec4f>;
@group(0) @binding(4) var<storage, read> csr_row_ptr: array<u32>;
@group(0) @binding(5) var<storage, read> csr_col_idx: array<u32>;
@group(0) @binding(6) var<storage, read> csr_values: array<f32>;
@group(0) @binding(7) var<storage, read> diag: array<f32>;

fn read_csr_block(offset: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(csr_values[offset + 0u], csr_values[offset + 1u], csr_values[offset + 2u]),
        vec3f(csr_values[offset + 3u], csr_values[offset + 4u], csr_values[offset + 5u]),
        vec3f(csr_values[offset + 6u], csr_values[offset + 7u], csr_values[offset + 8u]),
    );
}

fn read_diag_block(node: u32) -> mat3x3f {
    let base = node * 9u;
    return mat3x3f(
        vec3f(diag[base + 0u], diag[base + 1u], diag[base + 2u]),
        vec3f(diag[base + 3u], diag[base + 4u], diag[base + 5u]),
        vec3f(diag[base + 6u], diag[base + 7u], diag[base + 8u]),
    );
}

fn mat3_mul_vec3(m: mat3x3f, v: vec3f) -> vec3f {
    return vec3f(
        dot(m[0], v),
        dot(m[1], v),
        dot(m[2], v),
    );
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.node_count) {
        return;
    }

    let dt2 = params.dt * params.dt;
    let pi = cg_p[id].xyz;
    let mi = mass[id].x;

    // Mass term: M * p
    var result = mi * pi;

    // H * p: off-diagonal CSR
    var hp = vec3f(0.0, 0.0, 0.0);

    let row_start = csr_row_ptr[id];
    let row_end = csr_row_ptr[id + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col = csr_col_idx[idx];
        let block = read_csr_block(idx * 9u);
        let pj = cg_p[col].xyz;
        hp = hp + mat3_mul_vec3(block, pj);
    }

    // H * p: diagonal
    let diag_block = read_diag_block(id);
    hp = hp + mat3_mul_vec3(diag_block, pi);

    // Ap = M*p - dt^2 * H*p
    result = result - dt2 * hp;

    cg_ap[id] = vec4f(result, 0.0);
}
