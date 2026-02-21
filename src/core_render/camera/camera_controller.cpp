#include "core_render/camera/camera_controller.h"
#include "core_render/camera/camera.h"
#include "core_platform/input.h"

using namespace mps::util;
using namespace mps::platform;

namespace mps {
namespace render {

CameraController::CameraController(Camera& camera, const CameraControllerConfig& config)
    : camera_(camera)
    , config_(config) {
}

void CameraController::Update(float32 dt) {
    auto& input = InputManager::GetInstance();

    bool ctrl_held = input.IsKeyHeld(Key::LeftControl) || input.IsKeyHeld(Key::RightControl);

    if (input.IsMouseButtonHeld(MouseButton::Middle)) {
        auto delta = input.GetMouseDelta();
        if (ctrl_held) {
            // Ctrl + middle mouse = orbit
            camera_.Orbit(-delta.x * config_.orbit_speed, delta.y * config_.orbit_speed);
        } else {
            // Middle mouse = pan
            camera_.Pan(-delta.x * config_.pan_speed, delta.y * config_.pan_speed);
        }
    }

    // Scroll = zoom
    auto scroll = input.GetMouseScroll();
    if (scroll.y != 0.0f) {
        camera_.Zoom(scroll.y * config_.zoom_speed);
    }
}

}  // namespace render
}  // namespace mps
