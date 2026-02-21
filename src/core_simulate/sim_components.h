#pragma once

#include "core_util/types.h"

namespace mps {
namespace simulate {

// Generic particle simulation components — 16-byte aligned POD, maps to vec4<f32> in WGSL.
// These are ECS components registered with DeviceDB for GPU mirroring.

struct SimPosition {
    float32 x = 0.0f;
    float32 y = 0.0f;
    float32 z = 0.0f;
    float32 w = 0.0f;
};

struct SimVelocity {
    float32 vx = 0.0f;
    float32 vy = 0.0f;
    float32 vz = 0.0f;
    float32 w = 0.0f;
};

struct SimMass {
    float32 mass = 1.0f;
    float32 inv_mass = 1.0f;  // 0 = pinned
};

}  // namespace simulate
}  // namespace mps
