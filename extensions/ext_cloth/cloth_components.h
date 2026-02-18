#pragma once

#include "core_util/types.h"

namespace ext_cloth {

using namespace mps;

// 16-byte aligned POD — maps to vec4<f32> in WGSL
struct ClothPosition {
    float32 x = 0.0f;
    float32 y = 0.0f;
    float32 z = 0.0f;
    float32 w = 0.0f;
};

struct ClothVelocity {
    float32 vx = 0.0f;
    float32 vy = 0.0f;
    float32 vz = 0.0f;
    float32 w = 0.0f;
};

struct ClothMass {
    float32 mass = 1.0f;
    float32 inv_mass = 1.0f;  // 0 = pinned
    float32 pad0 = 0.0f;
    float32 pad1 = 0.0f;
};

}  // namespace ext_cloth
