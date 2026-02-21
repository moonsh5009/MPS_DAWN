// Normal computation parameters uniform — 16 bytes.
//
// Layout:
//   [0..3]   node_count, face_count, pad0, pad1   (u32 x4)

struct NormalParams {
    node_count: u32,
    face_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: NormalParams;
