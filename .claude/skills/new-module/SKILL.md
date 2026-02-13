---
name: new-module
description: Create a new C++ module with cross-platform structure, CMakeLists.txt, and all required files
argument-hint: "[module_name]"
---

# New Module Workflow

Create a complete module under `src/<module_name>/` following MPS_DAWN conventions.

## Input

- `module_name`: e.g. `core_gpu`, `core_render`
- Derive: namespace = second part (e.g. `gpu`, `render`), interface prefix from purpose

## Steps

### 1. Create directory

```
src/<module_name>/
```

### 2. Write interface header — `<name>.h`

```cpp
#pragma once
#include "core_util/types.h"
#include <memory>
#include <string>

namespace mps::<namespace> {

struct <Name>Config {
    std::string title = "Default";
    util::uint32 width = 1280;
    util::uint32 height = 720;
};

class I<Name> {
public:
    virtual ~I<Name>() = default;
    virtual bool Initialize(const <Name>Config& config) = 0;
    static std::unique_ptr<I<Name>> Create();
};

}  // namespace mps::<namespace>
```

### 3. Write factory — `<name>.cpp`

```cpp
#include "<module_name>/<name>.h"

#ifdef __EMSCRIPTEN__
#include "<module_name>/<name>_wasm.h"
#else
#include "<module_name>/<name>_native.h"
#endif

namespace mps::<namespace> {

std::unique_ptr<I<Name>> I<Name>::Create() {
#ifdef __EMSCRIPTEN__
    return std::make_unique<<Name>Wasm>();
#else
    return std::make_unique<<Name>Native>();
#endif
}

}  // namespace mps::<namespace>
```

### 4. Write native implementation — `<name>_native.h` + `<name>_native.cpp`

Header:
```cpp
#pragma once
#include "<module_name>/<name>.h"

namespace mps::<namespace> {

class <Name>Native : public I<Name> {
public:
    ~<Name>Native() override = default;
    bool Initialize(const <Name>Config& config) override;
};

}  // namespace mps::<namespace>
```

Implementation:
```cpp
#include "<module_name>/<name>_native.h"
#include "core_util/logger.h"

namespace mps::<namespace> {

bool <Name>Native::Initialize(const <Name>Config& config) {
    util::LogInfo("[<Name>Native] Initialize");
    return true;
}

}  // namespace mps::<namespace>
```

### 5. Write WASM implementation — `<name>_wasm.h` + `<name>_wasm.cpp`

Same structure as native, with `<Name>Wasm` class name and `[<Name>Wasm]` log prefix.

### 6. Write `CMakeLists.txt`

```cmake
add_library(<module_name> STATIC
    <name>.cpp
)

if(EMSCRIPTEN)
    target_sources(<module_name> PRIVATE <name>_wasm.cpp)
else()
    target_sources(<module_name> PRIVATE <name>_native.cpp)
endif()

set_target_properties(<module_name> PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(<module_name> PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(<module_name> PUBLIC mps::core_util)

# Add native-only dependencies here:
# if(NOT EMSCRIPTEN)
#     target_link_libraries(<module_name> PUBLIC some_native_lib)
# endif()

add_library(mps::<module_name> ALIAS <module_name>)
```

### 7. Register in `src/CMakeLists.txt`

Add `add_subdirectory(<module_name>)` in dependency order (after its dependencies, before its dependents).

### 8. Link from executable

In the top-level executable target, add `mps::<module_name>` to `target_link_libraries`.

## Checklist

- [ ] All files use `#pragma once`
- [ ] Project headers before STL headers
- [ ] Uses `util::` types (never raw `int`/`float`)
- [ ] PascalCase classes/methods, snake_case_ members
- [ ] No indentation inside namespace blocks
- [ ] 4 spaces, no tabs
- [ ] Logging via `LogInfo`/`LogError`
