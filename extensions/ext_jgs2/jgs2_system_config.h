#pragma once

#include "core_util/types.h"
#include "core_database/entity.h"

namespace ext_jgs2 {

using namespace mps;

// Host-only configuration component for the JGS2 solver.
struct JGS2SystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;

    uint32 iterations         = 10;     // Block Jacobi iterations per timestep
    bool enable_correction    = false;  // Phase 2: Schur complement correction

    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};

    uint32 mesh_entity        = database::kInvalidEntity;
    uint32 padding[3]         = {};
};

}  // namespace ext_jgs2
