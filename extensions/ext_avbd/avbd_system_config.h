#pragma once

#include "core_util/types.h"
#include "core_database/entity.h"

namespace ext_avbd {

using namespace mps;
using namespace mps::database;

// Host-only configuration component for an AVBD (Vertex Block Descent) system.
// References a mesh entity and constraint entities for term discovery.
struct AVBDSystemConfig {
    uint32 avbd_iterations    = 10;
    uint32 mesh_entity        = kInvalidEntity;
    uint32 constraint_count   = 0;
    float32 al_gamma          = 0.99f;     // AL warmstart penalty decay factor
    Entity constraint_entities[4] = {kInvalidEntity, kInvalidEntity, kInvalidEntity, kInvalidEntity};
    float32 al_beta           = 100000.0f; // AL penalty ramp rate per iteration
    // Total: 36 bytes
};

}  // namespace ext_avbd
