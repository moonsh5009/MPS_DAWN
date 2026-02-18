// CG convergence check: if rr_new < tolerance, set converged flag and zero indirect dispatch
// Dispatch: 1 workgroup (single thread)
//
// Scalar buffer layout:
//   [0] rr       - dot(r,r)
//   [1] pAp      - dot(p,Ap)
//   [2] rr_new   - dot(r_new, r_new)
//   [3] alpha    - rr / pAp
//   [4] beta     - rr_new / rr
//   [5] converged - 0=running, 1=converged

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
@group(0) @binding(1) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    let rr_new = scalars[2];
    if (rr_new < params.cg_tolerance) {
        scalars[5] = 1.0;   // converged flag
        indirect[0] = 0u;   // zero workgroup count -> skip per-node passes
    }
}
