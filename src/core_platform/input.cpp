#include "core_platform/input.h"
#include "core_util/logger.h"

using namespace mps::util;

namespace mps {
namespace platform {

InputManager::InputManager() {
    mouse_button_states_.fill(InputState::Released);
    prev_mouse_button_states_.fill(InputState::Released);
}

InputManager::~InputManager() = default;

void InputManager::Update() {
    // Update previous states
    prev_key_states_ = key_states_;
    prev_mouse_button_states_ = mouse_button_states_;

    // Update key states (Pressed -> Held, Released remains Released)
    for (auto& [key, state] : key_states_) {
        if (state == InputState::Pressed) {
            state = InputState::Held;
        }
    }

    // Update mouse button states
    for (size_t i = 0; i < mouse_button_states_.size(); ++i) {
        if (mouse_button_states_[i] == InputState::Pressed) {
            mouse_button_states_[i] = InputState::Held;
        }
    }

    // Calculate mouse delta
    mouse_delta_ = mouse_position_ - prev_mouse_position_;
    prev_mouse_position_ = mouse_position_;

    // Scroll: consume pending (accumulated since last frame), reset pending
    mouse_scroll_ = pending_scroll_;
    pending_scroll_ = {0.0f, 0.0f};
}

void InputManager::SetKeyState(Key key, bool pressed) {
    auto it = key_states_.find(key);
    if (pressed) {
        // Only set to Pressed if it wasn't already held
        if (it == key_states_.end() || it->second == InputState::Released) {
            key_states_[key] = InputState::Pressed;
        }
    } else {
        key_states_[key] = InputState::Released;
    }
}

bool InputManager::IsKeyPressed(Key key) const {
    auto it = key_states_.find(key);
    return it != key_states_.end() && it->second == InputState::Pressed;
}

bool InputManager::IsKeyHeld(Key key) const {
    auto it = key_states_.find(key);
    return it != key_states_.end() &&
           (it->second == InputState::Pressed || it->second == InputState::Held);
}

bool InputManager::IsKeyReleased(Key key) const {
    auto it = prev_key_states_.find(key);
    auto it_current = key_states_.find(key);

    // Released this frame if it was held last frame and is released now
    return it != prev_key_states_.end() &&
           (it->second == InputState::Pressed || it->second == InputState::Held) &&
           (it_current == key_states_.end() || it_current->second == InputState::Released);
}

void InputManager::SetMouseButtonState(MouseButton button, bool pressed) {
    size_t index = static_cast<size_t>(button);
    if (index >= mouse_button_states_.size()) {
        return;
    }

    if (pressed) {
        // Only set to Pressed if it wasn't already held
        if (mouse_button_states_[index] == InputState::Released) {
            mouse_button_states_[index] = InputState::Pressed;
        }
    } else {
        mouse_button_states_[index] = InputState::Released;
    }
}

bool InputManager::IsMouseButtonPressed(MouseButton button) const {
    size_t index = static_cast<size_t>(button);
    if (index >= mouse_button_states_.size()) {
        return false;
    }
    return mouse_button_states_[index] == InputState::Pressed;
}

bool InputManager::IsMouseButtonHeld(MouseButton button) const {
    size_t index = static_cast<size_t>(button);
    if (index >= mouse_button_states_.size()) {
        return false;
    }
    return mouse_button_states_[index] == InputState::Pressed ||
           mouse_button_states_[index] == InputState::Held;
}

bool InputManager::IsMouseButtonReleased(MouseButton button) const {
    size_t index = static_cast<size_t>(button);
    if (index >= mouse_button_states_.size()) {
        return false;
    }

    return (prev_mouse_button_states_[index] == InputState::Pressed ||
            prev_mouse_button_states_[index] == InputState::Held) &&
           mouse_button_states_[index] == InputState::Released;
}

void InputManager::SetMousePosition(float32 x, float32 y) {
    mouse_position_ = {x, y};
}

void InputManager::SetMouseScroll(float32 x, float32 y) {
    // Direct set (used by native GLFW callback during PollEvents)
    pending_scroll_ = {x, y};
}

void InputManager::AccumulateMouseScroll(float32 x, float32 y) {
    // Accumulate (used by WASM async callbacks between frames)
    pending_scroll_.x += x;
    pending_scroll_.y += y;
}

InputManager& InputManager::GetInstance() {
    static InputManager instance;
    return instance;
}

}  // namespace platform
}  // namespace mps
