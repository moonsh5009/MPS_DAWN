---
name: style
description: C++20 code style and formatting guide for writing MPS_DAWN code — file layout, formatting, const-correctness, initialization, modern C++ idioms
---

# C++20 Code Style & Formatting Guide

A proactive writing guide for new MPS_DAWN code. Use this **before** writing code; use `/review` **after** to check for violations.

> **Note:** Some existing code uses older patterns (nested namespace declarations). New code should follow this guide; refactor legacy patterns when touching those files.

## 1. File Layout

### Header file (`.h`)

```cpp
#pragma once

#include "core_util/types.h"       // Project headers first
#include <string>                   // Then STL
#include <memory>

// Forward declarations for external types
struct GLFWwindow;

namespace mps::platform {           // C++17 nested namespace

// Brief description of the class
class MyClass {
public:
    // ...
private:
    // ...
};

}  // namespace mps::platform
```

> Source pattern: `window.h`, `window_native.h`

### Implementation file (`.cpp`)

```cpp
#include "core_platform/window_native.h"  // Own header first
#include "core_platform/input.h"          // Project headers
#include "core_util/logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>                   // Third-party

using namespace mps::util;                // OK in .cpp, NEVER in .h

namespace mps::platform {                 // C++17 nested namespace

// Implementations here (no indentation inside namespace)

}  // namespace mps::platform
```

> Source pattern: `window_native.cpp:1-22`

### Closing namespace comments

Always annotate closing braces: `}  // namespace mps::platform` (two spaces before `//`).

## 2. Include Order

**`.cpp` files:** own header → project headers → third-party → STL

**`.h` files:** project headers → STL

Within each group, order alphabetically. Separate groups with a blank line.

```cpp
// window_native.cpp
#include "core_platform/window_native.h"   // 1. Own header
#include "core_platform/input.h"           // 2. Project headers
#include "core_util/logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>                    // 3. Third-party
```

> Source: `window_native.cpp:1-16`

## 3. Formatting Rules

### Indentation

- 4 spaces, no tabs
- No indentation inside namespace blocks

### Multi-line function arguments

Opening paren on same line, each argument indented, closing paren aligned or with last arg:

```cpp
window_ = glfwCreateWindow(
    static_cast<int>(config.width),
    static_cast<int>(config.height),
    config.title.c_str(),
    config.fullscreen ? glfwGetPrimaryMonitor() : nullptr,
    nullptr
);
```

> Source: `window_native.cpp:38-44`

### Multi-line function calls

Same pattern for chained or nested calls:

```cpp
InputManager::GetInstance().SetMousePosition(
    static_cast<float32>(xpos),
    static_cast<float32>(ypos)
);
```

> Source: `window_native.cpp:204-207`

### Blank lines

- Single blank line between methods
- No double blank lines anywhere
- No blank line after opening brace or before closing brace
- Blank line before comment sections: `// Section name`

### Braces

- Opening brace on same line: `class Foo {`, `if (x) {`
- Always use braces for control flow (no braceless `if`/`for`/`while`)

## 4. Const-Correctness & `[[nodiscard]]`

### Getter methods — always `const`, prefer `[[nodiscard]]`

Mark getters with `[[nodiscard]]` so callers cannot silently discard the result:

```cpp
[[nodiscard]] bool ShouldClose() const;
[[nodiscard]] bool IsMinimized() const;
[[nodiscard]] bool IsFocused() const;
[[nodiscard]] uint32 GetWidth() const;
[[nodiscard]] float32 GetAspectRatio() const;
```

> Baseline: `window_native.h:22-28`

### Factory methods — always `[[nodiscard]]`

```cpp
[[nodiscard]] static std::unique_ptr<IWindow> Create();
```

> Baseline: `window.h:50`

### Return by `const&` for member strings

```cpp
[[nodiscard]] const std::string& GetTitle() const;
```

> Baseline: `window.h:38`

### Parameters

| Type | Passing | Example |
|------|---------|---------|
| Non-trivial (struct, string) | `const Type&` | `const WindowConfig& config` |
| Read-only string (no storage) | `std::string_view` | `std::string_view name` |
| Primitive (`bool`, numeric) | by value | `bool fullscreen`, `float32 x` |

Use `std::string_view` when the function only reads the string and does not store it. Use `const std::string&` when the function stores or forwards it.

> Baseline: `window.h:25, 41-43`, `input.h:81`

## 5. Initialization & `constexpr`

### Default member initializers in header

Always initialize members at declaration — never leave them uninitialized:

```cpp
GLFWwindow* window_ = nullptr;
WindowConfig config_;
bool is_focused_ = true;
```

> Source: `window_native.h:50-52`

### Brace initialization for aggregates

```cpp
util::vec2 mouse_position_ = {0.0f, 0.0f};
util::vec2 prev_mouse_position_ = {0.0f, 0.0f};
util::vec2 mouse_delta_ = {0.0f, 0.0f};
util::vec2 mouse_scroll_ = {0.0f, 0.0f};
```

> Source: `input.h:102-107`

### Config structs with defaults

```cpp
struct WindowConfig {
    std::string title = "MPS_DAWN";
    uint32 width = 1280;
    uint32 height = 720;
    bool resizable = true;
    bool fullscreen = false;
};
```

> Source: `window.h:12-17`

### `constexpr` for compile-time constants and simple functions

Prefer `constexpr` over `const` for values known at compile time:

```cpp
constexpr float32 PI = 3.14159265358979323846f;
constexpr float32 TWO_PI = 2.0f * PI;
constexpr float32 DEG_TO_RAD = PI / 180.0f;
constexpr float32 RAD_TO_DEG = 180.0f / PI;
```

Use `constexpr` functions for computations that can be evaluated at compile time:

```cpp
constexpr float32 Radians(float32 degrees) {
    return degrees * DEG_TO_RAD;
}

constexpr float32 Degrees(float32 radians) {
    return radians * RAD_TO_DEG;
}
```

> Baseline: `math.h:38-42` (constants already `constexpr`; utility functions can be upgraded)

### `= default` and `= delete`

Use `= default` for trivial constructors/destructors. Mark `noexcept` where applicable:

```cpp
Logger() = default;
~Logger() = default;
```

Delete all four for non-copyable / non-movable types:

```cpp
Logger(const Logger&) = delete;
Logger& operator=(const Logger&) = delete;
Logger(Logger&&) = delete;
Logger& operator=(Logger&&) = delete;
```

> Source: `logger.h:61-68`

## 6. Enum Style

Always `enum class` with explicit underlying type from project types:

```cpp
enum class Key : uint16 {
    Unknown = 0,
    A = 65, B = 66, C = 67,
    // ...
};

enum class MouseButton : uint8 {
    Left = 0,
    Right = 1,
    Middle = 2,
};
```

> Source: `input.h:12-41`, `input.h:44-50`

For simple enums without explicit values, underlying type is optional:

```cpp
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};
```

> Source: `logger.h:10-15`

## 7. Type Aliases

Use private `using` aliases (never `typedef`), placed at top of private section:

```cpp
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start_time_;
    TimePoint stop_time_;
    bool is_running_;
```

> Source: `timer.h:35-41`

## 8. Template Formatting

`template<...>` on its own line, function signature on the next:

```cpp
template<typename... Args>
void Debug(Args&&... args) {
    if (min_level_ <= LogLevel::Debug) {
        Log(LogLevel::Debug, Format(std::forward<Args>(args)...));
    }
}
```

Fold expressions (C++17) for parameter packs:

```cpp
template<typename... Args>
std::string Format(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}
```

> Source: `logger.h:32-37`, `logger.h:73-78`

### C++20 Concepts (when constraints are needed)

Prefer concepts over SFINAE or unconstrained templates:

```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
constexpr T Clamp(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}
```

Use `requires` clause for ad-hoc constraints:

```cpp
template<typename T>
    requires std::is_enum_v<T>
constexpr auto ToUnderlying(T value) noexcept {
    return static_cast<std::underlying_type_t<T>>(value);
}
```

## 9. `auto` & Structured Bindings

### `auto` usage

Use `auto` when the type is obvious from context or overly verbose:

```cpp
// Good — type is clear from the right-hand side
auto* win = static_cast<WindowNative*>(glfwGetWindowUserPointer(window));
auto it = key_states_.find(key);
auto surface = wgpuInstanceCreateSurface(instance, &surfaceDesc);

// Bad — type is not obvious, spell it out
auto x = GetValue();           // What type is this?
uint32 width = GetWidth(); // Explicit is clearer here
```

### Structured bindings (C++17)

Prefer structured bindings for map iteration and pair/tuple unpacking:

```cpp
// Good — clear and concise
for (auto& [key, state] : key_states_) {
    if (state == InputState::Pressed) {
        state = InputState::Held;
    }
}

// Instead of
for (auto& pair : key_states_) {
    if (pair.second == InputState::Pressed) {
        pair.second = InputState::Held;
    }
}
```

> Source: `input.cpp:22-26`

## 10. Unused Parameters & Standard Attributes

### Unused parameters — compiler warning suppression

Unused parameter warnings are disabled globally in CMake (`/wd4100` for MSVC, `-Wno-unused-parameter` for GCC/Clang). No annotation needed — leave parameter names as-is for readability:

```cpp
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Key mapped_key = static_cast<Key>(key);
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    InputManager::GetInstance().SetKeyState(mapped_key, pressed);
}
```

> Source: `window_native.cpp:187-192`

**Do NOT** use `(void)param` casts or `[[maybe_unused]]` for unused parameters.

### Standard attributes

| Attribute | Use case | Example |
|-----------|----------|---------|
| `[[nodiscard]]` | Getters, factory methods, error codes | `[[nodiscard]] bool Initialize(...)` |
| `[[likely]]` / `[[unlikely]]` | Hot-path branch hints (C++20) | `if (window_) [[likely]] { ... }` |
| `[[deprecated("use X")]]` | API migration | `[[deprecated("use NewFunc")]] void OldFunc();` |

## 11. `noexcept`

### When to use

| Context | Rule |
|---------|------|
| Move constructors / move assignment | Always `noexcept` (enables STL optimizations) |
| Destructors | Implicitly `noexcept`, but mark explicitly when custom |
| Swap functions | Always `noexcept` |
| Simple getters returning by value/ref | Use when body cannot throw |
| Functions calling C APIs (GLFW, WebGPU) | Generally safe to mark `noexcept` |

```cpp
class ResourceHandle {
public:
    ResourceHandle(ResourceHandle&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    ResourceHandle& operator=(ResourceHandle&& other) noexcept {
        if (this != &other) {
            Release();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] bool IsValid() const noexcept { return handle_ != nullptr; }

private:
    void* handle_ = nullptr;
};
```

### When NOT to use

- Functions that allocate memory (`std::string`, `std::vector` operations)
- Functions calling code that may throw
- When unsure — omit it; a wrong `noexcept` causes `std::terminate`

## 12. Inline Convenience Functions

### Free functions wrapping singleton access

```cpp
[[nodiscard]] inline bool IsKeyPressed(Key key) {
    return InputManager::GetInstance().IsKeyPressed(key);
}
[[nodiscard]] inline bool IsKeyHeld(Key key) {
    return InputManager::GetInstance().IsKeyHeld(key);
}
[[nodiscard]] inline bool IsKeyReleased(Key key) {
    return InputManager::GetInstance().IsKeyReleased(key);
}
```

> Baseline: `input.h:112-114`

### Template convenience functions

```cpp
template<typename... Args>
void LogInfo(Args&&... args) {
    Logger::GetInstance().Info(std::forward<Args>(args)...);
}

template<typename... Args>
void LogError(Args&&... args) {
    Logger::GetInstance().Error(std::forward<Args>(args)...);
}
```

> Source: `logger.h:89-99`
