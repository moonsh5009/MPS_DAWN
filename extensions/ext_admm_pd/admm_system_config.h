#pragma once

#include "core_util/types.h"
#include "core_database/entity.h"

namespace ext_admm_pd {

using namespace mps;

// Host-only configuration component for the ADMM PD solver.
struct ADMMSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;

    uint32 admm_iterations    = 20;    // Outer ADMM iterations
    uint32 cg_iterations      = 10;    // Inner CG iterations per ADMM step

    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};

    uint32 mesh_entity        = database::kInvalidEntity;
    uint32 padding[3]         = {};
};

}  // namespace ext_admm_pd
