// PD Spring LHS: Constant system matrix assembly (w * S^T * S)
// Dispatch: ceil(edge_count / 64) workgroups
//
// For each spring edge (a,b) with weight w = stiffness:
//   S = [I, -I] maps 2-node positions to relative displacement
//   S^T * S produces:
//     diag[a] += w * I_3
//     diag[b] += w * I_3
//     csr[a,b] += -w * I_3
//     csr[b,a] += -w * I_3
//
// Only diagonal entries [0],[4],[8] of each 3x3 block are written
// since all blocks are scalar multiples of the identity matrix.

#import "core_simulate/header/solver_params.wgsl"
#import "core_simulate/header/atomic_float.wgsl"

struct SpringEdge {
    n0: u32,
    n1: u32,
    rest_length: f32,
};

struct SpringParams {
    stiffness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> solver: SolverParams;
@group(0) @binding(1) var<storage, read> edges: array<SpringEdge>;
@group(0) @binding(2) var<storage, read_write> diag: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> csr_values: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> edge_csr_map: array<vec4u>;
@group(0) @binding(5) var<uniform> spring_params: SpringParams;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let eid = gid.x;
    if (eid >= solver.edge_count) {
        return;
    }

    let edge = edges[eid];
    let a = edge.n0;
    let b = edge.n1;
    let w = spring_params.stiffness;

    // Diagonal blocks: diag[a] += w * I_3, diag[b] += w * I_3
    let diag_a = a * 9u;
    let diag_b = b * 9u;

    atomicAddFloat(&diag[diag_a + 0u], w);
    atomicAddFloat(&diag[diag_a + 4u], w);
    atomicAddFloat(&diag[diag_a + 8u], w);

    atomicAddFloat(&diag[diag_b + 0u], w);
    atomicAddFloat(&diag[diag_b + 4u], w);
    atomicAddFloat(&diag[diag_b + 8u], w);

    // Off-diagonal CSR blocks: csr[a,b] += -w * I_3, csr[b,a] += -w * I_3
    let mapping = edge_csr_map[eid];
    let csr_ab = mapping.x * 9u;
    let csr_ba = mapping.y * 9u;

    atomicAddFloat(&csr_values[csr_ab + 0u], -w);
    atomicAddFloat(&csr_values[csr_ab + 4u], -w);
    atomicAddFloat(&csr_values[csr_ab + 8u], -w);

    atomicAddFloat(&csr_values[csr_ba + 0u], -w);
    atomicAddFloat(&csr_values[csr_ba + 4u], -w);
    atomicAddFloat(&csr_values[csr_ba + 8u], -w);
}
