#pragma once

#include "core_util/types.h"
#include "core_util/math.h"

namespace mps {
namespace simulate {

// Host-side singleton (stored in Database via SetSingleton)
struct GlobalPhysicsParams {
    float32 dt        = 1.0f / 60.0f;
    util::vec3 gravity = {0.0f, -9.81f, 0.0f};
    float32 damping   = 0.999f;
};

// GPU-side uniform (binding 0, managed by DeviceDB singleton)
struct alignas(16) PhysicsParamsGPU {
    float32 dt;
    float32 gravity_x, gravity_y, gravity_z;
    float32 damping, inv_dt, dt_sq, inv_dt_sq;
};

inline PhysicsParamsGPU ToGPU(const GlobalPhysicsParams& p) {
    return {p.dt, p.gravity.x, p.gravity.y, p.gravity.z,
            p.damping, 1.0f / p.dt, p.dt * p.dt, 1.0f / (p.dt * p.dt)};
}

}  // namespace simulate
}  // namespace mps
