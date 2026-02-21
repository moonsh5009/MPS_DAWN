#pragma once

#include "core_util/types.h"

namespace ext_newton {

using namespace mps;

// Host-only configuration component for a Newton-Raphson system.
// References constraint entities by ID; not GPU-synced.
struct NewtonSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;

    uint32 newton_iterations  = 1;
    uint32 cg_max_iterations  = 30;
    float32 damping           = 0.999f;
    float32 cg_tolerance      = 1e-6f;

    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};

    uint32 padding[3]         = {};
    // Total: 64 bytes
};

}  // namespace ext_newton
