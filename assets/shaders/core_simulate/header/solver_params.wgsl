// Solver parameters uniform — layout-compatible with SolverParams C++ struct.
// Binding 1: per-solver mesh topology counts and CG configuration.
//
// Layout (32 bytes = 2 x vec4):
//   [0..3]   node_count, edge_count, face_count, cg_max_iter   (u32 x4)
//   [4..7]   cg_tolerance, _pad0, _pad1, _pad2                 (f32 x4)

struct SolverParams {
    node_count: u32,
    edge_count: u32,
    face_count: u32,
    cg_max_iter: u32,
    cg_tolerance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
