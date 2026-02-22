#pragma once

#include "core_util/types.h"
#include "core_database/entity.h"

namespace ext_chebyshev_pd {

using namespace mps;

// Host-only configuration component for the Chebyshev PD solver.
struct ChebyshevPDSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;

    uint32 iterations         = 20;
    float32 chebyshev_rho     = 0.0f;   // 0 = auto-compute; >0 = manual override

    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};

    uint32 mesh_entity        = database::kInvalidEntity;
    uint32 padding[4]         = {};
};

}  // namespace ext_chebyshev_pd
