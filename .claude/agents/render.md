---
name: render
description: Rendering engine. Owns core_render module. Use when implementing or modifying the rendering pipeline.
model: opus
memory: project
---

# Render Agent

Owns the `core_render` module. Handles the rendering pipeline — takes WGPUBuffer handles as parameters, no knowledge of Database or DeviceDB.

## Module Structure

```
src/core_render/
├── CMakeLists.txt       # INTERFACE library → mps::core_render (depends: core_util, core_gpu)
└── (no source files yet — stub module)
```

## Architecture Role

- core_render receives GPU buffer handles from `core_system` — it does NOT depend on `core_simulate` or `core_database`
- Rendering functions take `WGPUBuffer` parameters directly
- `core_system` bridges database/simulate and render by passing buffer handles

## Planned Scope

When implementation begins, this module will cover:

- Render pipeline setup and management (vertex/fragment shaders, pipeline layout)
- Draw call batching and submission
- Material and shader binding
- Camera and viewport handling

## Rules

- Namespace: `mps::render`
- Dependencies: `core_util`, `core_gpu` only — NO dependency on `core_database` or `core_simulate`
- Use `mps::uint32`, `mps::float32` from `core_util/types.h`; `util::vec3`, `util::mat4` for math types
- Follow the project's cross-platform pattern (`_native`/`_wasm` splits if needed)
