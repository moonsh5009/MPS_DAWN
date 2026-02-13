---
name: module-management
description: Module architecture, C++20 coding standards, cross-platform patterns. Owns core_util and core_platform modules. Use for refactoring, style questions, or architecture decisions.
model: opus
memory: project
---

# Module Management Agent

Owns `core_util` and `core_platform` modules. Maintains architecture, C++20 coding standards, and cross-platform patterns for MPS_DAWN. Module overview and namespaces are in CLAUDE.md.

> **Skills**: `/new-module` (create module), `/type-ref` (type system), `/review` (code review checklist)

## Module Rules

- Static library + CMake alias (`mps::module_name`) per module
- Dependencies flow downward only, no circular deps
- Each module has its own `CMakeLists.txt`

## Cross-Platform Pattern

Interface + factory + separate `_native`/`_wasm` files:

```cpp
// window.h — abstract interface
class IWindow {
public:
    virtual ~IWindow() = default;
    virtual bool Initialize(const WindowConfig& config) = 0;
    static std::unique_ptr<IWindow> Create();  // factory
};

// window.cpp — factory selects platform
std::unique_ptr<IWindow> IWindow::Create() {
#ifdef __EMSCRIPTEN__
    return std::make_unique<WindowWasm>();
#else
    return std::make_unique<WindowNative>();
#endif
}
```

**File naming**: `module.h` (interface), `module_native.cpp/h`, `module_wasm.cpp/h`
**Platform detection**: `#ifdef __EMSCRIPTEN__` (WASM) | `_WIN32` | `__linux__` | `__APPLE__`

## Naming Conventions

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

## Code Style

- **Indentation**: 4 spaces, no tabs; no indentation inside namespace blocks
- **Braces**: same line (`class Foo {`, `if (x) {`), always use for control flow
- **Spacing**: `if (cond)` not `if(cond)`, `Function()` not `Function ()`
- **Comments**: `//` only (use `/* */` only to disable code blocks); prefer `TODO:`, `FIXME:`, `NOTE:`
- **Comments**: explain WHY, not WHAT

## Memory & Error Handling

- Stack allocation > heap; `std::unique_ptr` for ownership; `std::shared_ptr` only when truly shared
- Raw pointers only for non-owning refs; always init to `nullptr`
- Return `bool` for success/failure; `nullptr` from factory on failure
- Log with `LogDebug/Info/Warning/Error(...)` from `core_util/logger.h` (variadic, concatenates args)

## Rules Summary

**DO**: Use project type aliases (`/type-ref`), factory pattern for platform objects, const-correctness, pass-by-const-ref, move semantics
**DON'T**: Raw `new`/`delete`, circular deps, platform APIs in interfaces, `using namespace` in headers, tabs
