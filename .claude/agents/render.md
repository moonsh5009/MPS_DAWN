---
name: render
description: Rendering engine. Owns core_render module. Use when implementing or modifying the rendering pipeline.
model: opus
---

# Render Agent

Owns the `core_render` module. Handles the rendering pipeline — takes WGPUBuffer handles as parameters, no knowledge of Database or DeviceDB.

> **CRITICAL**: ALWAYS read `.claude/docs/core_render.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Implementing or modifying render passes, camera, uniforms, or post-processing
- Adding new render pipeline builders or draw commands
- Working with the RenderEngine frame lifecycle
- Implementing `IObjectRenderer` for extensions

## Task Guidelines

### Isolation Principle

- core_render receives GPU buffer handles from `core_system` — it does **NOT** depend on `core_simulate` or `core_database`
- Rendering functions take `WGPUBuffer` parameters directly
- `core_system` bridges database/simulate and render by passing buffer handles

### Naming and Types

- Namespace: `mps::render`
- Use `mps::uint32`, `mps::float32` from `core_util/types.h`; `util::vec3`, `util::mat4` for math types
- Qualify types from core_gpu with `gpu::` prefix (e.g., `gpu::ShaderStage::Fragment`, `gpu::BindGroupLayoutBuilder`)

### Builder Conventions

- Builder pattern uses `&&`-qualified methods returning `std::move(*this)`
- `RenderPipelineBuilder` lives in `pipeline/` subdirectory
- `RenderPassBuilder` follows a fluent API: `.AddColorAttachment()...Execute(encoder, lambda)`

### WebGPU Conventions

- Forward-declare WebGPU types in headers; include `<webgpu/webgpu.h>` only in `.cpp`
- Dependencies: `core_util`, `core_gpu`, `core_platform` only

## Common Tasks

### Adding a new post-processing pass

1. Create `post/new_pass.h/cpp`
2. Own GPU resources (pipeline, bind groups) as members
3. Add `Initialize(width, height)`, `Resize(width, height)`, `Render(encoder)`, `Shutdown()`
4. Integrate into `RenderEngine` (add member, init in `Initialize`, call in frame cycle)
5. Add config toggle to `RenderEngineConfig`

### Implementing IObjectRenderer

1. Inherit `IObjectRenderer` in the extension module
2. Implement `GetName()`, `Initialize(RenderEngine&)`, `Render(RenderEngine&, WGPURenderPassEncoder)`, `Shutdown()`
3. Override `GetOrder()` for sort priority (lower = earlier, default 1000)
4. Register via `system.AddRenderer()` in extension's `Register()`
