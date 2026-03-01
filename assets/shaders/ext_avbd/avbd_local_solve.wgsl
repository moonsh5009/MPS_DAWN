// AVBD local solve with inertia: reads term gradient + Hessian, adds inertia,
// solves 3×3 via cofactor inverse, updates q, then zeroes gradient/hessian
// for the next color/iteration.
// Pinned nodes (inv_mass <= 0) are skipped entirely.
// Dispatch: ceil(color_vertex_count / 64) per color group.

#import "core_simulate/header/physics_params.wgsl"
#import "ext_avbd/header/vbd_params.wgsl"

struct SimMass {
    mass: f32,
    inv_mass: f32,
};

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<uniform> color_params: VBDColorParams;
@group(0) @binding(2) var<storage, read_write> q: array<vec4f>;
@group(0) @binding(3) var<storage, read> s: array<vec4f>;
@group(0) @binding(4) var<storage, read> mass: array<SimMass>;
@group(0) @binding(5) var<storage, read_write> gradient: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> hessian: array<vec4f>;
@group(0) @binding(7) var<storage, read> vertex_order: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let local_idx = gid.x;
    if (local_idx >= color_params.color_vertex_count) {
        return;
    }

    let vi = vertex_order[color_params.color_offset + local_idx];
    let m = mass[vi];

    // Pinned node: zero gradient/hessian and skip
    if (m.inv_mass <= 0.0) {
        gradient[vi] = vec4f(0.0);
        hessian[vi * 3u + 0u] = vec4f(0.0);
        hessian[vi * 3u + 1u] = vec4f(0.0);
        hessian[vi * 3u + 2u] = vec4f(0.0);
        return;
    }

    // Read term contributions
    var g = gradient[vi].xyz;
    var row0 = hessian[vi * 3u + 0u].xyz;
    var row1 = hessian[vi * 3u + 1u].xyz;
    var row2 = hessian[vi * 3u + 2u].xyz;

    // Add inertia: gradient += w*(q-s), hessian_diag += w
    let w = m.mass * physics.inv_dt_sq;
    let diff = q[vi].xyz - s[vi].xyz;
    g += w * diff;
    row0.x += w;
    row1.y += w;
    row2.z += w;

    // Cofactor 3×3 inverse
    let h00 = row0.x; let h01 = row0.y; let h02 = row0.z;
    let h10 = row1.x; let h11 = row1.y; let h12 = row1.z;
    let h20 = row2.x; let h21 = row2.y; let h22 = row2.z;

    let c11 = h11 * h22 - h12 * h21;
    let c12 = -(h10 * h22 - h12 * h20);
    let c13 = h10 * h21 - h11 * h20;
    let c21 = -(h01 * h22 - h02 * h21);
    let c22 = h00 * h22 - h02 * h20;
    let c23 = -(h00 * h21 - h01 * h20);
    let c31 = h01 * h12 - h02 * h11;
    let c32 = -(h00 * h12 - h02 * h10);
    let c33 = h00 * h11 - h01 * h10;

    let det = h00 * c11 + h01 * c12 + h02 * c13;
    if (abs(det) < 1e-20) {
        // Singular — zero and skip
        gradient[vi] = vec4f(0.0);
        hessian[vi * 3u + 0u] = vec4f(0.0);
        hessian[vi * 3u + 1u] = vec4f(0.0);
        hessian[vi * 3u + 2u] = vec4f(0.0);
        return;
    }

    let inv_det = 1.0 / det;

    // dx = -H⁻¹ * g (adjugate = transposed cofactor)
    let dx = vec3f(
        -inv_det * (c11 * g.x + c21 * g.y + c31 * g.z),
        -inv_det * (c12 * g.x + c22 * g.y + c32 * g.z),
        -inv_det * (c13 * g.x + c23 * g.y + c33 * g.z),
    );

    q[vi] = q[vi] + vec4f(dx, 0.0);

    // Zero gradient/hessian for next color/iteration
    gradient[vi] = vec4f(0.0);
    hessian[vi * 3u + 0u] = vec4f(0.0);
    hessian[vi * 3u + 1u] = vec4f(0.0);
    hessian[vi * 3u + 2u] = vec4f(0.0);
}
