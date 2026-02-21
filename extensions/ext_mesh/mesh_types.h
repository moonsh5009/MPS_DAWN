#pragma once

#include "core_util/types.h"

namespace ext_mesh {

using namespace mps;

// Triangle face topology (GPU-compatible, read as array<vec4u> in shaders)
struct alignas(16) MeshFace {
    uint32 n0 = 0;
    uint32 n1 = 0;
    uint32 n2 = 0;
};

// Fixed/pinned vertex record (host-only)
struct FixedVertex {
    uint32 vertex_index = 0;        // local index within mesh
    float32 original_mass = 0.0f;   // saved for restoration when unpinned
    float32 original_inv_mass = 0.0f;
};

}  // namespace ext_mesh
