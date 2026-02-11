# MPS_DAWN Project Instructions

This file contains project-wide instructions that apply to all Claude Code sessions working on this project.

## Build and Testing Policy

### Build Tests
- **DO NOT** perform build tests unless explicitly requested by the user
- Only execute build commands when the user specifically asks to:
  - "빌드해줘" / "build it"
  - "테스트해줘" / "test it"
  - "실행해줘" / "run it"
- When making changes to CMakeLists.txt or build configuration:
  - Explain the changes
  - DO NOT automatically verify with a build
  - Wait for user confirmation before building

### Example
❌ **Incorrect**:
```
I've updated CMakeLists.txt. Let me build to verify...
[runs cmake --build]
```

✅ **Correct**:
```
I've updated CMakeLists.txt with the new output directory settings.
The changes are complete. Let me know if you'd like me to test the build.
```

---

## Project Understanding

### Reference Documentation
- **ALWAYS** refer to the markdown files in `.claude/guide/` to understand the project's current state
- Key guide files:
  - `DEVELOPMENT_SETUP.md` - Complete development environment setup and build configuration
  - (More guides will be added as the project progresses)
- These guides contain:
  - Current project structure
  - Build system configuration
  - Completed setup steps
  - Known issues and solutions
- Read these files before making suggestions or changes to understand what's already been done

---

## Project Overview

### Identity
- **MPS_DAWN** = "My(Moon) Physics Simulation"
- High-performance physics simulation framework built on **WebGPU (Dawn)**
- Target platforms: Windows x64 (native) + WebAssembly (WASM)
- C++20 standard

### Core Purpose
- GPU-accelerated physics simulation using WebGPU
- Cross-platform rendering and compute
- Modular architecture for extensibility

---

## Architecture Principles

### Module Structure
The project follows a layered architecture with 6 core modules:

```
src/
├── core_util/       # Foundation (no dependencies)
├── core_gpu/        # WebGPU abstraction (depends on: core_util)
├── core_platform/   # Platform abstraction (depends on: core_util)
├── core_database/   # Data management (depends on: core_util)
├── core_render/     # Rendering engine (depends on: core_util, core_gpu)
└── core_simulate/   # Simulation management (depends on: core_util, core_gpu, core_database)
```

### Dependency Rules
- **core_util** is the foundation layer with NO dependencies
- Higher-level modules may depend on lower-level modules
- **NO circular dependencies** allowed between modules
- Dependencies must be explicit in CMakeLists.txt

### Namespace Organization
- Root namespace: `mps`
- Module namespaces: `mps::util`, `mps::gpu`, `mps::platform`, `mps::database`, `mps::render`, `mps::simulate`

---

## Coding Standards

### C++ Standards
- Use **C++20** standard features
- Enable standard compliance: `CXX_STANDARD 20`, `CXX_STANDARD_REQUIRED ON`

### CMake Patterns
- Use **ALIAS** pattern for library linking:
  ```cmake
  add_library(core_util STATIC ...)
  add_library(mps::core_util ALIAS core_util)
  ```
- Link using namespace-style: `target_link_libraries(target PRIVATE mps::core_util)`
- Include directories should use `${CMAKE_SOURCE_DIR}/src` for project headers

### File Organization
- Header files: `module_name/file.h`
- Source files: `module_name/file.cpp`
- Each module has its own CMakeLists.txt
- Public headers go in module root, private in subdirectories if needed

### Naming Conventions
- Types: PascalCase (e.g., `Logger`, `Timer`)
- Functions: PascalCase for public API (e.g., `GetInstance()`)
- Variables: snake_case with trailing underscore for members (e.g., `min_level_`)
- Namespaces: lowercase (e.g., `mps::util`)

---

## Key Dependencies

### Core Libraries
- **Dawn** - Google's WebGPU implementation (native builds only)
- **GLM** - OpenGL Mathematics library (header-only, via git submodule)
  - Location: `third_party/glm/`
  - Used through `mps::util` math wrapper

### Platform-Specific
- Native builds: Link `webgpu_dawn`
- WASM builds: Use Emscripten's WebGPU support (no Dawn)

---

## Module Guidelines

### Creating a New Module
1. Create directory: `src/module_name/`
2. Create CMakeLists.txt following this template:
   ```cmake
   # module_name - Brief description

   add_library(module_name STATIC
       # source files
   )

   set_target_properties(module_name PROPERTIES
       CXX_STANDARD 20
       CXX_STANDARD_REQUIRED ON
   )

   target_include_directories(module_name PUBLIC
       ${CMAKE_SOURCE_DIR}/src
   )

   target_link_libraries(module_name PUBLIC
       # dependencies (e.g., mps::core_util)
   )

   add_library(mps::module_name ALIAS module_name)
   ```
3. Add to parent CMakeLists.txt: `add_subdirectory(module_name)`
4. Update ARCHITECTURE.md with module documentation

### Module Design Rules
- Each module should have a single, clear responsibility
- Keep public API minimal and well-documented
- Use forward declarations to minimize header dependencies
- Consider GPU data layout (Structure-of-Arrays) for performance

---

## Additional Instructions
(Instructions will be added here as the project progresses)
