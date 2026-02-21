---
name: review
description: Code review against MPS_DAWN project standards — naming, style, memory, const-correctness
---

# Code Review Checklist

Review code against MPS_DAWN C++20 standards. Check each category and report violations.

## 1. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Class / Struct | PascalCase | `WindowNative`, `WindowConfig` |
| Interface | `I` + PascalCase | `IWindow`, `IGpuContext` |
| Method / Function | PascalCase | `Initialize()`, `ShouldClose()` |
| Private member | snake_case + `_` | `window_`, `is_focused_` |
| Local / Param | snake_case | `actual_width`, `config` |
| Type alias | lowercase | `uint32`, `vec3`, `mat4` |
| Constant | UPPER_SNAKE_CASE | `PI`, `DEG_TO_RAD` |
| Namespace | lowercase | `mps::util`, `mps::platform` |

## 2. Code Style

- [ ] 4 spaces indentation, no tabs
- [ ] No indentation inside namespace blocks
- [ ] Braces on same line (`class Foo {`, `if (x) {`)
- [ ] Always use braces for control flow (no braceless if/for/while)
- [ ] Space after keywords: `if (cond)` not `if(cond)`
- [ ] No space before parens: `Function()` not `Function ()`
- [ ] `//` comments only (use `/* */` only to disable code blocks)
- [ ] Comments explain WHY, not WHAT
- [ ] Prefer `TODO:`, `FIXME:`, `NOTE:` prefixes
- [ ] No `(void)param` casts or `[[maybe_unused]]` for unused parameters (suppressed by compiler flags)

## 3. Headers & Includes

- [ ] `#pragma once` on every header
- [ ] Include order: own header first (in .cpp), then project headers, then STL
- [ ] No `using namespace` in headers (OK in .cpp)

## 4. Type System

- [ ] Primitives (`uint32`, `float32`, etc.) from `mps` namespace — no `util::` prefix needed
- [ ] Math types: `util::vec3`, `util::mat4`, etc. (remain in `mps::util`)
- [ ] Never use raw `int`/`float`/`size_t` (except at C-library API boundaries)
- [ ] Constants from `core_util/math.h` (e.g., `PI`, `DEG_TO_RAD`)

## 5. Memory & Error Handling

- [ ] Prefer stack allocation over heap
- [ ] `std::unique_ptr` for ownership; `std::shared_ptr` only when truly shared
- [ ] Raw pointers only for non-owning references, initialized to `nullptr`
- [ ] No raw `new`/`delete`
- [ ] Return `bool` for success/failure; `nullptr` from factory on failure
- [ ] Logging via `LogDebug/Info/Warning/Error(...)` from `core_util/logger.h`

## 6. Cross-Platform

- [ ] Platform code in separate `_native`/`_wasm` files
- [ ] No platform-specific APIs in interface headers
- [ ] Platform detection via `#ifdef __EMSCRIPTEN__`

## 7. General

- [ ] const-correctness (const refs, const methods)
- [ ] Pass by const reference for non-trivial types
- [ ] Move semantics where appropriate
- [ ] No circular dependencies between modules
- [ ] No tabs anywhere

## Common Patterns

### Config Struct (defaults + override)

```cpp
struct WindowConfig {
    std::string title = "MPS_DAWN";
    uint32 width = 1280;
    uint32 height = 720;
    bool resizable = true;
};
```

### Singleton

```cpp
class Logger {
public:
    static Logger& GetInstance() { static Logger inst; return inst; }
private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};
```

### C Library Callbacks (GLFW example)

```cpp
glfwSetWindowUserPointer(window_, this);
static void Callback(GLFWwindow* w, ...) {
    auto* self = static_cast<MyClass*>(glfwGetWindowUserPointer(w));
    if (self) self->HandleEvent(...);
}
```

## 8. Build Verification

After code review, verify the project compiles on both platforms:

### Native (Debug)

```bash
cmake -B build && cmake --build build
```

- [ ] CMake configures without errors
- [ ] Build completes without errors (warnings from third-party libs OK)
- [ ] Executable runs: `build\bin\x64\Debug\mps_dawn.exe`

### WASM (Debug)

```bash
# Windows (requires emsdk + ninja):
cmd //c "C:\emsdk\emsdk_env.bat >nul 2>&1 && set PATH=<ninja-path>;%PATH% && emcmake cmake -B build-wasm && cmake --build build-wasm"
```

- [ ] CMake configures with Emscripten toolchain
- [ ] Build completes without errors
- [ ] Output: `build-wasm/bin/Debug/mps_dawn.html`

### Common Build Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Duplicate `glfw` target | Dawn includes its own GLFW | Remove separate `third_party/glfw`; Dawn provides `glfw` target |
| `GLM_GTX_component_wise` error | GLM experimental extension | Add `#define GLM_ENABLE_EXPERIMENTAL` before GLM `gtx` includes |
| `webgpu/webgpu.h` not found | Missing Dawn link in CMake | Link `webgpu_dawn` for native builds |
| `-sUSE_WEBGPU=1` invalid | Emscripten API changed | Use `--use-port=emdawnwebgpu` instead |
| `WGPUStringView` type mismatch | emdawnwebgpu uses StringView | Use `{"string", length}` initializer |

## Review Output Format

For each violation found, report:
```
[CATEGORY] file:line — description of violation
```

Summarize: total files reviewed, violations found per category, overall pass/fail.
