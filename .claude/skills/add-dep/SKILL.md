---
name: add-dep
description: Add a third-party dependency as a git submodule with CMake integration
argument-hint: "[library_name] [repository_url]"
---

# Add Dependency Workflow

Add a third-party library to MPS_DAWN as a git submodule.

## Input

- `library_name`: e.g. `imgui`, `stb`
- `repository_url`: git repository URL
- Platform scope: **all** | **native-only** | **wasm-only**

## Steps

### 1. Add git submodule

```bash
git submodule add <repository_url> third_party/<library_name>
```

### 2. Update root `CMakeLists.txt`

Add the subdirectory with platform guard if needed:

**All platforms:**
```cmake
add_subdirectory(third_party/<library_name>)
```

**Native-only** (like Dawn, GLFW):
```cmake
if(NOT EMSCRIPTEN)
    add_subdirectory(third_party/<library_name>)
endif()
```

Place alongside existing third-party entries. Follow existing patterns for cache variables if the library requires configuration:

```cmake
# Example: Dawn configuration pattern
set(SOME_OPTION ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
```

### 3. Link in module's `CMakeLists.txt`

```cmake
target_link_libraries(<module_name> PUBLIC <library_cmake_target>)
```

With platform guard if native-only:
```cmake
if(NOT EMSCRIPTEN)
    target_link_libraries(<module_name> PUBLIC <library_cmake_target>)
endif()
```

### 4. Update `.gitmodules`

Verify the submodule entry was added correctly:
```
[submodule "third_party/<library_name>"]
    path = third_party/<library_name>
    url = <repository_url>
```

### 5. Update CLAUDE.md

Add the new dependency to the Third-Party Dependencies line in the Architecture section.

## Checklist

- [ ] Submodule added under `third_party/`
- [ ] Root CMakeLists.txt updated with correct platform guard
- [ ] Module CMakeLists.txt links the library
- [ ] `.gitmodules` has correct entry
- [ ] CLAUDE.md Third-Party Dependencies updated
- [ ] Build succeeds: `cmake -B build && cmake --build build`
