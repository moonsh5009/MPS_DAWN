#pragma once

#include "core_util/types.h"
#include "core_database/entity.h"

namespace ext_pd {

using namespace mps;

// Host-only configuration component for a Projective Dynamics system.
// References constraint entities by ID; not GPU-synced.
struct PDSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;

    uint32 iterations         = 20;     // Wang 2015 single fused loop iterations
    float32 chebyshev_rho     = 0.0f;   // 0 = auto-compute from LHS; >0 = manual override

    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};

    uint32 mesh_entity        = database::kInvalidEntity;  // kInvalidEntity = global mode, valid entity = scoped
    uint32 padding[4]         = {};
    // Total: 64 bytes
};

}  // namespace ext_pd
