---
name: gpu
description: WebGPU abstraction via Dawn (native) and Emscripten (WASM). Owns core_gpu module. Use for GPU context setup, surface creation, shader management, or WebGPU API questions.
model: opus
---

# GPU Agent

Owns the `core_gpu` module. Handles WebGPU integration, Dawn vs Emscripten, resource management.

> **CRITICAL**: ALWAYS read `.claude/docs/core_gpu.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Implementing or modifying GPU resource classes (buffers, textures, shaders, samplers)
- Adding or modifying builders (bind group, pipeline layout, compute pipeline)
- Working with GPUCore lifecycle (initialization, surface creation, shutdown)
- Handling Dawn-specific or Emscripten-specific WebGPU behavior
- Shader loading or `#import` directive logic

## Task Guidelines

### Include Conventions

- **Never** include `<webgpu/webgpu.h>` in headers — use forward declarations instead
- **Never** include `dawn/dawn_proc.h` — monolithic `webgpu_dawn` handles procs internally
- Forward-declare WebGPU handle types in headers:
  ```cpp
  struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;
  ```

### WASM Guards

- Dawn-specific code: guard with `#ifndef __EMSCRIPTEN__`
- Native async pattern: `WGPUCallbackMode_WaitAnyOnly` + `wgpuInstanceWaitAny`
- WASM async pattern: `WGPUCallbackMode_AllowProcessEvents` + `ProcessEvents()` polling

### Dawn API Notes

- Use `WGPU_<TYPE>_INIT` macros for struct initialization (e.g., `WGPU_INSTANCE_DESCRIPTOR_INIT`)
- Strings use `WGPUStringView{data, length}`, not `const char*`
- Adapter info: `wgpuAdapterGetInfo()` + `wgpuAdapterInfoFreeMembers()`
- Must enable `WGPUInstanceFeatureName_TimedWaitAny` in instance features for native sync waits
- `DAWN_FORCE_SYSTEM_COMPONENT_LOAD ON` in root CMakeLists.txt — loads system DLLs from System32
- **VertexFormat enum alignment**: `gpu_types.h` values MUST match Dawn's `WGPUVertexFormat`
- **Uncaptured error callback**: Set on `WGPUDeviceDescriptor::uncapturedErrorCallbackInfo` during device request

### Builder Conventions

- Fluent rvalue-reference API: `Builder().Method().Build()`
- All builders return `GPUHandle<Tag>` wrapped types
- Surface creation: `GPUCore::CreateSurface()` handles platform-specific logic internally

### Resource Conventions

- All resource classes follow RAII: move-only, direct construction, config structs with defaults, throw `GPUException` on failure
- `GPUBuffer<T>`: split design — `GPUBufferCore` (non-template, .cpp) + `GPUBuffer<T>` (template, header-only)
- Buffer internals: 1.5x geometric growth, 16-byte alignment, `CopySrc` auto-added for Grow support
- Shaders: use `ShaderLoader::CreateModule()` for file-loaded WGSL, or `GPUShader` for inline code
- When referencing builders/surface/shader from other modules, use `gpu::` prefix

## Common Tasks

### Adding a new builder

1. Create `new_builder.h/cpp` in `src/core_gpu/`
2. Follow fluent `&&`-qualified method pattern returning `std::move(*this)`
3. `Build()` returns a `GPUHandle<Tag>` wrapped type
4. Add to `CMakeLists.txt`
