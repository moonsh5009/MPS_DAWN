// VBD per-color parameters uniform.
// Binding 0 in local solve shader: identifies which color group to process.
//
// Layout (16 bytes = 1 x vec4):
//   [0..3]   color_offset, color_vertex_count, _pad0, _pad1   (u32 x4)

struct VBDColorParams {
    color_offset: u32,
    color_vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
};
