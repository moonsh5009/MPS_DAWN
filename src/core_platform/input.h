#pragma once

#include "core_util/types.h"
#include "core_util/math.h"
#include <array>
#include <unordered_map>

namespace mps {
namespace platform {

// Key codes (matches GLFW key codes for convenience)
enum class Key : uint16 {
    Unknown = 0,

    // Alphanumeric keys
    A = 65, B = 66, C = 67, D = 68, E = 69, F = 70, G = 71, H = 72,
    I = 73, J = 74, K = 75, L = 76, M = 77, N = 78, O = 79, P = 80,
    Q = 81, R = 82, S = 83, T = 84, U = 85, V = 86, W = 87, X = 88,
    Y = 89, Z = 90,

    Num0 = 48, Num1 = 49, Num2 = 50, Num3 = 51, Num4 = 52,
    Num5 = 53, Num6 = 54, Num7 = 55, Num8 = 56, Num9 = 57,

    // Function keys
    F1 = 290, F2 = 291, F3 = 292, F4 = 293, F5 = 294, F6 = 295,
    F7 = 296, F8 = 297, F9 = 298, F10 = 299, F11 = 300, F12 = 301,

    // Arrow keys
    Left = 263, Right = 262, Up = 265, Down = 264,

    // Control keys
    Space = 32, Enter = 257, Tab = 258, Backspace = 259, Escape = 256,
    LeftShift = 340, RightShift = 344,
    LeftControl = 341, RightControl = 345,
    LeftAlt = 342, RightAlt = 346,

    // Special keys
    CapsLock = 280, NumLock = 282, ScrollLock = 281,
    Insert = 260, Delete = 261, Home = 268, End = 269,
    PageUp = 266, PageDown = 267,
};

// Mouse button codes
enum class MouseButton : uint8 {
    Left = 0,
    Right = 1,
    Middle = 2,
    Button4 = 3,
    Button5 = 4,
};

// Input state for a single key/button
enum class InputState : uint8 {
    Released = 0,  // Not pressed
    Pressed = 1,   // Just pressed this frame
    Held = 2,      // Held down
};

// Input manager
class InputManager {
public:
    InputManager();
    ~InputManager();

    // Update input state (call once per frame)
    void Update();

    // Keyboard input
    void SetKeyState(Key key, bool pressed);
    bool IsKeyPressed(Key key) const;   // True only on the frame it was pressed
    bool IsKeyHeld(Key key) const;      // True while held down
    bool IsKeyReleased(Key key) const;  // True only on the frame it was released

    // Mouse input
    void SetMouseButtonState(MouseButton button, bool pressed);
    bool IsMouseButtonPressed(MouseButton button) const;
    bool IsMouseButtonHeld(MouseButton button) const;
    bool IsMouseButtonReleased(MouseButton button) const;

    // Mouse position
    void SetMousePosition(float32 x, float32 y);
    util::vec2 GetMousePosition() const { return mouse_position_; }
    util::vec2 GetMouseDelta() const { return mouse_delta_; }

    // Mouse scroll (accumulated between frames, consumed on Update)
    void SetMouseScroll(float32 x, float32 y);
    util::vec2 GetMouseScroll() const { return mouse_scroll_; }
    void AccumulateMouseScroll(float32 x, float32 y);

    // Singleton access
    static InputManager& GetInstance();

private:
    // Keyboard state
    std::unordered_map<Key, InputState> key_states_;
    std::unordered_map<Key, InputState> prev_key_states_;

    // Mouse state
    std::array<InputState, 5> mouse_button_states_;
    std::array<InputState, 5> prev_mouse_button_states_;

    // Mouse position and delta
    util::vec2 mouse_position_ = {0.0f, 0.0f};
    util::vec2 prev_mouse_position_ = {0.0f, 0.0f};
    util::vec2 mouse_delta_ = {0.0f, 0.0f};

    // Mouse scroll (double-buffered for async WASM events)
    util::vec2 mouse_scroll_ = {0.0f, 0.0f};
    util::vec2 pending_scroll_ = {0.0f, 0.0f};

};

// Convenience functions
inline bool IsKeyPressed(Key key) { return InputManager::GetInstance().IsKeyPressed(key); }
inline bool IsKeyHeld(Key key) { return InputManager::GetInstance().IsKeyHeld(key); }
inline bool IsKeyReleased(Key key) { return InputManager::GetInstance().IsKeyReleased(key); }

inline bool IsMouseButtonPressed(MouseButton button) {
    return InputManager::GetInstance().IsMouseButtonPressed(button);
}
inline bool IsMouseButtonHeld(MouseButton button) {
    return InputManager::GetInstance().IsMouseButtonHeld(button);
}
inline bool IsMouseButtonReleased(MouseButton button) {
    return InputManager::GetInstance().IsMouseButtonReleased(button);
}

inline util::vec2 GetMousePosition() { return InputManager::GetInstance().GetMousePosition(); }
inline util::vec2 GetMouseDelta() { return InputManager::GetInstance().GetMouseDelta(); }
inline util::vec2 GetMouseScroll() { return InputManager::GetInstance().GetMouseScroll(); }

}  // namespace platform
}  // namespace mps
