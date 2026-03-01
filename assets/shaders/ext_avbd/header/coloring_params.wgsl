// Graph coloring parameters uniform
// Binding 0 in all graph coloring shaders.

struct ColoringParams {
    node_count: u32,
    edge_count: u32,
    scan_count: u32,
    _pad0: u32,
};
