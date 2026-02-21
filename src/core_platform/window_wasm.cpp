#include "core_platform/window_wasm.h"
#include "core_platform/input.h"
#include "core_util/logger.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

#include <string>
#include <unordered_map>

using namespace mps::util;

namespace mps {
namespace platform {

WindowWasm::WindowWasm() = default;

WindowWasm::~WindowWasm() {
    Shutdown();
}

bool WindowWasm::Initialize(const WindowConfig& config) {
#ifdef __EMSCRIPTEN__
    config_ = config;
    current_width_ = config.width;
    current_height_ = config.height;

    // Set canvas element size (drawing buffer resolution)
    emscripten_set_canvas_element_size("#canvas", current_width_, current_height_);

    // Set CSS display size to match
    emscripten_set_element_css_size("#canvas", current_width_, current_height_);

    // Set document title
    emscripten_run_script(("document.title = '" + config.title + "';").c_str());

    // Register input event callbacks
    RegisterInputCallbacks();

    LogInfo("WASM Window initialized: ", current_width_, "x", current_height_);
    return true;
#else
    LogError("WindowWasm can only be used in Emscripten builds");
    return false;
#endif
}

void WindowWasm::Shutdown() {
    // Nothing to clean up for WASM
    LogInfo("WASM Window shutdown");
}

void WindowWasm::PollEvents() {
#ifdef __EMSCRIPTEN__
    // Update canvas size in case it changed
    double width, height;
    emscripten_get_element_css_size("#canvas", &width, &height);
    current_width_ = static_cast<uint32>(width);
    current_height_ = static_cast<uint32>(height);
#endif
}

bool WindowWasm::ShouldClose() const {
    return should_close_;
}

bool WindowWasm::IsMinimized() const {
    return false;
}

bool WindowWasm::IsFocused() const {
    return is_focused_;
}

uint32 WindowWasm::GetWidth() const {
    return current_width_;
}

uint32 WindowWasm::GetHeight() const {
    return current_height_;
}

float32 WindowWasm::GetAspectRatio() const {
    return current_height_ > 0 ?
        static_cast<float32>(current_width_) / static_cast<float32>(current_height_) : 0.0f;
}

const std::string& WindowWasm::GetTitle() const {
    return config_.title;
}

void WindowWasm::SetTitle(const std::string& title) {
#ifdef __EMSCRIPTEN__
    config_.title = title;
    emscripten_run_script(("document.title = '" + title + "';").c_str());
#endif
}

void WindowWasm::SetSize(uint32 width, uint32 height) {
#ifdef __EMSCRIPTEN__
    config_.width = width;
    config_.height = height;
    current_width_ = width;
    current_height_ = height;
    emscripten_set_canvas_element_size("#canvas", width, height);
#endif
}

void WindowWasm::SetFullscreen(bool fullscreen) {
#ifdef __EMSCRIPTEN__
    config_.fullscreen = fullscreen;
    if (fullscreen) {
        EmscriptenFullscreenStrategy strategy = {};
        strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
        strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_HIDEF;
        strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
        emscripten_request_fullscreen_strategy("#canvas", 1, &strategy);
    } else {
        emscripten_exit_fullscreen();
    }
#endif
}

void* WindowWasm::GetNativeWindowHandle() const {
    return nullptr;
}

void* WindowWasm::GetNativeDisplayHandle() const {
    return nullptr;
}

// =============================================================================
// DOM KeyboardEvent.code → GLFW-compatible Key mapping
// Uses e->code string (modern, non-deprecated) instead of e->keyCode
// =============================================================================

Key WindowWasm::MapDOMKeyCode(unsigned long /*unused*/, unsigned long /*unused*/) {
    return Key::Unknown;  // Legacy — use MapDOMCode instead
}

static Key MapDOMCode(const char* code) {
    static const std::unordered_map<std::string, Key> map = {
        // Letters
        {"KeyA", Key::A}, {"KeyB", Key::B}, {"KeyC", Key::C}, {"KeyD", Key::D},
        {"KeyE", Key::E}, {"KeyF", Key::F}, {"KeyG", Key::G}, {"KeyH", Key::H},
        {"KeyI", Key::I}, {"KeyJ", Key::J}, {"KeyK", Key::K}, {"KeyL", Key::L},
        {"KeyM", Key::M}, {"KeyN", Key::N}, {"KeyO", Key::O}, {"KeyP", Key::P},
        {"KeyQ", Key::Q}, {"KeyR", Key::R}, {"KeyS", Key::S}, {"KeyT", Key::T},
        {"KeyU", Key::U}, {"KeyV", Key::V}, {"KeyW", Key::W}, {"KeyX", Key::X},
        {"KeyY", Key::Y}, {"KeyZ", Key::Z},
        // Digits
        {"Digit0", Key::Num0}, {"Digit1", Key::Num1}, {"Digit2", Key::Num2},
        {"Digit3", Key::Num3}, {"Digit4", Key::Num4}, {"Digit5", Key::Num5},
        {"Digit6", Key::Num6}, {"Digit7", Key::Num7}, {"Digit8", Key::Num8},
        {"Digit9", Key::Num9},
        // Function keys
        {"F1", Key::F1}, {"F2", Key::F2}, {"F3", Key::F3}, {"F4", Key::F4},
        {"F5", Key::F5}, {"F6", Key::F6}, {"F7", Key::F7}, {"F8", Key::F8},
        {"F9", Key::F9}, {"F10", Key::F10}, {"F11", Key::F11}, {"F12", Key::F12},
        // Arrow keys
        {"ArrowLeft", Key::Left}, {"ArrowRight", Key::Right},
        {"ArrowUp", Key::Up}, {"ArrowDown", Key::Down},
        // Control keys
        {"Space", Key::Space}, {"Enter", Key::Enter}, {"Tab", Key::Tab},
        {"Backspace", Key::Backspace}, {"Escape", Key::Escape},
        // Modifiers (left/right)
        {"ShiftLeft", Key::LeftShift}, {"ShiftRight", Key::RightShift},
        {"ControlLeft", Key::LeftControl}, {"ControlRight", Key::RightControl},
        {"AltLeft", Key::LeftAlt}, {"AltRight", Key::RightAlt},
        // Special
        {"CapsLock", Key::CapsLock}, {"NumLock", Key::NumLock},
        {"ScrollLock", Key::ScrollLock},
        {"Insert", Key::Insert}, {"Delete", Key::Delete},
        {"Home", Key::Home}, {"End", Key::End},
        {"PageUp", Key::PageUp}, {"PageDown", Key::PageDown},
    };

    auto it = map.find(code);
    return it != map.end() ? it->second : Key::Unknown;
}

// DOM mouse button: 0=Left, 1=Middle, 2=Right
// Our enum:         Left=0, Right=1, Middle=2
MouseButton WindowWasm::MapDOMMouseButton(unsigned short button) {
    switch (button) {
        case 0:  return MouseButton::Left;
        case 1:  return MouseButton::Middle;
        case 2:  return MouseButton::Right;
        case 3:  return MouseButton::Button4;
        case 4:  return MouseButton::Button5;
        default: return MouseButton::Left;
    }
}

// =============================================================================
// Emscripten HTML5 event callbacks
// =============================================================================

void WindowWasm::RegisterInputCallbacks() {
#ifdef __EMSCRIPTEN__
    // Make canvas focusable and auto-focus it for keyboard events
    emscripten_run_script(
        "var c = document.getElementById('canvas');"
        "c.tabIndex = 0;"
        "c.style.outline = 'none';"
        "c.focus();"
    );

    // Keyboard: register on window using e->code (modern, non-deprecated)
    emscripten_set_keydown_callback(
        EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int type, const EmscriptenKeyboardEvent* e, void*) -> EM_BOOL {
            Key key = MapDOMCode(e->code);
            if (key != Key::Unknown) {
                InputManager::GetInstance().SetKeyState(key, true);
            }
            // Prevent default for game keys to avoid page scroll
            if (key == Key::Space || key == Key::Tab ||
                key == Key::Left || key == Key::Right ||
                key == Key::Up || key == Key::Down) {
                return 1;
            }
            return 0;
        });

    emscripten_set_keyup_callback(
        EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int type, const EmscriptenKeyboardEvent* e, void*) -> EM_BOOL {
            Key key = MapDOMCode(e->code);
            if (key != Key::Unknown) {
                InputManager::GetInstance().SetKeyState(key, false);
            }
            return 0;
        });

    // Mouse buttons: register on canvas
    emscripten_set_mousedown_callback(
        "#canvas", nullptr, true,
        [](int type, const EmscriptenMouseEvent* e, void*) -> EM_BOOL {
            MouseButton btn = MapDOMMouseButton(e->button);
            InputManager::GetInstance().SetMouseButtonState(btn, true);
            return 1;
        });

    emscripten_set_mouseup_callback(
        "#canvas", nullptr, true,
        [](int type, const EmscriptenMouseEvent* e, void*) -> EM_BOOL {
            MouseButton btn = MapDOMMouseButton(e->button);
            InputManager::GetInstance().SetMouseButtonState(btn, false);
            return 1;
        });

    // Mouse move: register on canvas
    emscripten_set_mousemove_callback(
        "#canvas", nullptr, true,
        [](int type, const EmscriptenMouseEvent* e, void*) -> EM_BOOL {
            InputManager::GetInstance().SetMousePosition(
                static_cast<float32>(e->targetX),
                static_cast<float32>(e->targetY));
            return 0;
        });

    // Mouse wheel: register on canvas (preventDefault to avoid page scroll)
    emscripten_set_wheel_callback(
        "#canvas", nullptr, true,
        [](int type, const EmscriptenWheelEvent* e, void*) -> EM_BOOL {
            // Accumulate: WASM events fire between frames, not during PollEvents
            // DOM deltaY positive = scroll down, negate for GLFW-like convention
            InputManager::GetInstance().AccumulateMouseScroll(
                static_cast<float32>(-e->deltaX * 0.01),
                static_cast<float32>(-e->deltaY * 0.01));
            return 1;  // preventDefault
        });

    // Focus tracking (use lambda with userData for instance access)
    emscripten_set_focus_callback(
        "#canvas", this, true,
        [](int type, const EmscriptenFocusEvent*, void* ud) -> EM_BOOL {
            static_cast<WindowWasm*>(ud)->is_focused_ = true;
            return 0;
        });

    emscripten_set_blur_callback(
        "#canvas", this, true,
        [](int type, const EmscriptenFocusEvent*, void* ud) -> EM_BOOL {
            static_cast<WindowWasm*>(ud)->is_focused_ = false;
            return 0;
        });

    // Prevent right-click context menu on canvas using JavaScript
    emscripten_run_script(
        "document.getElementById('canvas').addEventListener('contextmenu', "
        "(e) => e.preventDefault(), false);"
    );
#endif
}

}  // namespace platform
}  // namespace mps
