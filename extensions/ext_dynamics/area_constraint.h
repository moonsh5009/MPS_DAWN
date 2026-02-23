#pragma once

#include "core_util/types.h"

namespace ext_dynamics {

using namespace mps;

// Host-only configuration component for area preservation constraint.
// Attached to a constraint entity referenced by NewtonSystemConfig.
// Triangle topology is stored as ArrayStorage<AreaTriangle> on the same entity.
struct AreaConstraintData {
    float32 stretch_stiffness = 1.0f;  // area preservation (bulk modulus k)
    float32 shear_stiffness = 0.0f;    // ARAP shear modulus (μ)
};

}  // namespace ext_dynamics
