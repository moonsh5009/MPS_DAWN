#pragma once

#include "core_util/types.h"

namespace ext_dynamics {

using namespace mps;

// Host-only configuration component for spring constraint.
// Attached to a constraint entity referenced by NewtonSystemConfig.
// Edge topology is stored as ArrayStorage<SpringEdge> on the same entity.
struct SpringConstraintData {
    float32 stiffness = 500.0f;
};

}  // namespace ext_dynamics
