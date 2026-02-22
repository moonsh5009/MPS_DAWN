#pragma once

#include "core_util/types.h"

namespace mps {
namespace simulate {

// Per-solver uniform buffer (binding 1).
// Contains mesh topology counts and CG configuration.
struct alignas(16) SolverParams {
    uint32 node_count   = 0;
    uint32 edge_count   = 0;
    uint32 face_count   = 0;
    uint32 cg_max_iter  = 30;
    float32 cg_tolerance = 1e-6f;
    float32 _pad0       = 0.0f;
    float32 _pad1       = 0.0f;
    float32 _pad2       = 0.0f;
};

}  // namespace simulate
}  // namespace mps
