// Accumulate spring forces and assemble Hessian blocks
// Dispatch: ceil(edge_count / 64) workgroups
//
// Per edge (a, b) with stiffness k and rest length L:
//   dx = pos[a] - pos[b], dist = |dx|, dir = dx / dist
//   Force:  f = -k * (dist - L) * dir
//   Cross Jacobian dfa/dxb (off-diagonal block):
//     J_ab = k * ((1 - L/dist) * I + (L/dist) * dir * dirT)
//   Self Jacobian dfa/dxa = -J_ab
//   Diagonal accumulation: diag_a += -J_ab = dfa/dxa
//
// The full Jacobian df/dx is negative semi-definite, ensuring
// the system matrix A = M - dt^2*J is always positive definite.
//
// edge_csr_mapping[e] = vec4u(csr_idx_ab, csr_idx_ba, diag_a, diag_b)

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

struct ClothEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
    stiffness: f32,
};

@group(0) @binding(0) var<uniform> params: ClothSimParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> edges: array<ClothEdge>;
@group(0) @binding(4) var<storage, read_write> csr_values: array<f32>;
@group(0) @binding(5) var<storage, read_write> diag_values: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> edge_csr_map: array<vec4u>;

fn atomicAddFloat(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old_val = atomicLoad(addr);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old_val) + val);
        let result = atomicCompareExchangeWeak(addr, old_val, new_val);
        if result.exchanged {
            break;
        }
        old_val = result.old_value;
    }
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= params.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;
    let k = edge.stiffness;
    let rest_len = edge.rest_length;

    let pa = positions[a].xyz;
    let pb = positions[b].xyz;
    let dx = pa - pb;
    let dist = length(dx);

    // Avoid division by zero for degenerate edges
    if (dist < 1e-8) {
        return;
    }

    let dir = dx / dist;

    // --- Spring force ---
    // f = -k * (dist - rest_len) * dir
    let f_spring = -k * (dist - rest_len) * dir;

    // Atomic add force to node a: force[a] += f_spring
    let base_a = a * 4u;
    atomicAddFloat(&forces[base_a + 0u], f_spring.x);
    atomicAddFloat(&forces[base_a + 1u], f_spring.y);
    atomicAddFloat(&forces[base_a + 2u], f_spring.z);

    // Atomic add force to node b: force[b] -= f_spring
    let base_b = b * 4u;
    atomicAddFloat(&forces[base_b + 0u], -f_spring.x);
    atomicAddFloat(&forces[base_b + 1u], -f_spring.y);
    atomicAddFloat(&forces[base_b + 2u], -f_spring.z);

    // --- Hessian assembly ---
    let ratio = rest_len / dist;
    let coeff_i = k * (1.0 - ratio);    // coefficient for identity part
    let coeff_d = k * ratio;            // coefficient for dir*dirT part

    // 3x3 block H_ab (row-major)
    var h: array<f32, 9>;
    h[0] = coeff_i + coeff_d * dir.x * dir.x;
    h[1] = coeff_d * dir.x * dir.y;
    h[2] = coeff_d * dir.x * dir.z;
    h[3] = coeff_d * dir.y * dir.x;
    h[4] = coeff_i + coeff_d * dir.y * dir.y;
    h[5] = coeff_d * dir.y * dir.z;
    h[6] = coeff_d * dir.z * dir.x;
    h[7] = coeff_d * dir.z * dir.y;
    h[8] = coeff_i + coeff_d * dir.z * dir.z;

    // Write off-diagonal CSR blocks (symmetric: H_ab = H_ba)
    let mapping = edge_csr_map[eid];
    let csr_ab = mapping.x * 9u;
    let csr_ba = mapping.y * 9u;

    for (var i = 0u; i < 9u; i = i + 1u) {
        csr_values[csr_ab + i] = h[i];
        csr_values[csr_ba + i] = h[i];
    }

    // Accumulate diagonal blocks: diag_a -= H_ab, diag_b -= H_ab
    let diag_a = a * 9u;
    let diag_b = b * 9u;

    for (var i = 0u; i < 9u; i = i + 1u) {
        atomicAddFloat(&diag_values[diag_a + i], -h[i]);
        atomicAddFloat(&diag_values[diag_b + i], -h[i]);
    }
}
