#pragma once

#include "core_util/types.h"

namespace ext_dynamics {

using namespace mps;

// Host-only configuration component for area preservation constraint.
// Attached to a constraint entity referenced by NewtonSystemConfig.
// Triangle topology is stored as ArrayStorage<AreaTriangle> on the same entity.
struct AreaConstraintData {
    float32 stiffness = 1.0f;
};

}  // namespace ext_dynamics
