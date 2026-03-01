// Augmented Lagrangian penalty parameters uniform.
// Used by warmstart_penalty and penalty_ramp shaders.
//
// Layout (16 bytes = 1 x vec4):
//   [0..3]   stiffness, gamma, beta, edge_count

struct ALParams {
    stiffness: f32,     // k (material stiffness, penalty cap)
    gamma: f32,         // warmstart decay factor
    beta: f32,          // penalty ramp rate per iteration
    edge_count: u32,
};
