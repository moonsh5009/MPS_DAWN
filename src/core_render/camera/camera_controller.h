#pragma once

#include "core_util/types.h"

namespace mps {
namespace render {

class Camera;

struct CameraControllerConfig {
    float32 orbit_speed = 0.005f;
    float32 pan_speed = 0.01f;
    float32 zoom_speed = 1.0f;
};

class CameraController {
public:
    CameraController(Camera& camera, const CameraControllerConfig& config = {});

    void Update(float32 dt);

private:
    Camera& camera_;
    CameraControllerConfig config_;
};

}  // namespace render
}  // namespace mps
