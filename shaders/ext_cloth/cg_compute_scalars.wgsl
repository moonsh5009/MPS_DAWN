// CG scalar computation
// Dispatch: 1 workgroup of 1 thread
//
// Scalar buffer layout (8 f32):
//   [0] rr       — current r dot r
//   [1] pAp      — p dot Ap
//   [2] rr_new   — new r dot r (after x,r update)
//   [3] alpha    — output: rr / pAp
//   [4] beta     — output: rr_new / rr
//
// Mode uniform (u32):
//   0: compute alpha = rr / pAp
//   1: compute beta = rr_new / rr, then advance rr = rr_new

struct ModeParams {
    mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(1) var<uniform> mode_params: ModeParams;

@compute @workgroup_size(1)
fn cs_main() {
    let mode = mode_params.mode;

    if (mode == 0u) {
        // Compute alpha = rr / pAp
        let rr = scalars[0];
        let pap = scalars[1];
        scalars[3] = rr / max(pap, 1e-30);
    } else {
        // Compute beta = rr_new / rr_old, advance rr
        let rr_old = scalars[0];
        let rr_new = scalars[2];
        scalars[4] = rr_new / max(rr_old, 1e-30);
        scalars[0] = rr_new;
    }
}
