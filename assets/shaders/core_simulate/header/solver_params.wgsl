// Solver parameters uniform — layout-compatible with ClothSimParams.
// Both structs can share the same GPU uniform buffer.
//
// Layout (48 bytes = 3 x vec4):
//   [0..3]   dt, gravity_x, gravity_y, gravity_z   (f32 x4)
//   [4..7]   node_count, edge_count, face_count, cg_max_iter   (u32 x4)
//   [8..11]  damping, cg_tolerance, pad0, pad1   (f32 x4)

struct SolverParams {
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
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: SolverParams;
