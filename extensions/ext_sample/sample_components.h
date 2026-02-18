#pragma once

#include "core_util/types.h"

namespace ext_sample {

using namespace mps;

struct SampleTransform {
    float32 x = 0.0f;
    float32 y = 0.0f;
    float32 z = 0.0f;
    float32 pad = 0.0f;
};

struct SampleVelocity {
    float32 vx = 0.0f;
    float32 vy = 0.0f;
    float32 vz = 0.0f;
    float32 pad = 0.0f;
};

}  // namespace ext_sample
