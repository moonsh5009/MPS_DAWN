#pragma once

#include "core_util/types.h"

namespace ext_newton {

using namespace mps;

// Host-only configuration component for gravity constraint.
// Attached to a constraint entity referenced by NewtonSystemConfig.
struct GravityConstraintData {
    float32 gx = 0.0f;
    float32 gy = -9.81f;
    float32 gz = 0.0f;
};

}  // namespace ext_newton
