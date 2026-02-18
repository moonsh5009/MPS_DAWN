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
├── CMakeLists.txt                          # STATIC library → mps::core_render (depends: core_util, core_gpu, core_platform)
├── render_types.h                          # Forward-declared WebGPU handles, render-specific enums
├── render_engine.h/cpp                     # Frame lifecycle orchestrator (BeginFrame/EndFrame)
├── pipeline/
│   └── render_pipeline_builder.h/cpp       # Fluent render pipeline builder
├── camera/
│   ├── camera.h/cpp                        # Orbit camera (view + projection matrices)
│   └── camera_controller.h/cpp             # Mouse-driven orbit/pan/zoom
├── uniform/
│   ├── camera_uniform.h/cpp                # Camera UBO sync to GPU
│   └── light_uniform.h/cpp                 # Directional light UBO
├── pass/
│   ├── render_pass_builder.h/cpp           # Render pass descriptor + execution
│   └── render_encoder.h/cpp                # Unified RenderPassEncoder/RenderBundleEncoder
├── target/
│   └── render_target.h/cpp                 # Resizable depth/color texture target
├── geometry/
│   └── draw_command.h/cpp                  # Mesh binding + draw batch
└── post/
    ├── fullscreen_quad.h/cpp               # Screen-space triangle utility
    ├── fxaa_pass.h/cpp                     # FXAA post-processing
    └── wboit_pass.h/cpp                    # Weighted blended OIT
```

## Architecture Role

- core_render receives GPU buffer handles from `core_system` — it does NOT depend on `core_simulate` or `core_database`
- Rendering functions take `WGPUBuffer` parameters directly
- `core_system` bridges database/simulate and render by passing buffer handles
- Uses `gpu::SurfaceManager`, `gpu::ShaderLoader`, `gpu::BindGroup*Builder` from core_gpu (qualify with `gpu::` prefix)

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `RenderEngine` | `render_engine.h` | Frame lifecycle: BeginFrame/EndFrame, sub-system orchestration |
| `RenderPipelineBuilder` | `pipeline/render_pipeline_builder.h` | Fluent render pipeline builder |
| `Camera` | `camera/camera.h` | Orbit camera with view/projection matrices |
| `CameraController` | `camera/camera_controller.h` | Mouse input → orbit/pan/zoom |
| `CameraUniform` | `uniform/camera_uniform.h` | Camera UBO (matrices, position, viewport) |
| `LightUniform` | `uniform/light_uniform.h` | Directional light UBO |
| `RenderPassBuilder` | `pass/render_pass_builder.h` | Render pass descriptor builder + execution |
| `RenderEncoder` | `pass/render_encoder.h` | Unified pass/bundle encoder wrapper |
| `RenderTarget` | `target/render_target.h` | Resizable GPU texture target |
| `DrawCommand` / `DrawList` | `geometry/draw_command.h` | Mesh binding + batched draw |
| `FXAAPass` | `post/fxaa_pass.h` | FXAA anti-aliasing post-process |
| `WBOITPass` | `post/wboit_pass.h` | Weighted blended OIT transparency |

## render_types.h

Render-specific enums (values match WebGPU):

- `CullMode`, `FrontFace` — rasterizer state
- `LoadOp`, `StoreOp` — attachment operations
- `BlendFactor`, `BlendOp` — blend state
- `BlendState`, `ClearColor` — aggregate structs

GPU-general enums (`ShaderStage`, `BindingType`, `VertexFormat`, `VertexStepMode`, `PrimitiveTopology`) live in `core_gpu/gpu_types.h`.

## Rules

- Namespace: `mps::render`
- Dependencies: `core_util`, `core_gpu`, `core_platform` — NO dependency on `core_database` or `core_simulate`
- Use `mps::uint32`, `mps::float32` from `core_util/types.h`; `util::vec3`, `util::mat4` for math types
- Qualify types moved to core_gpu with `gpu::` prefix (e.g., `gpu::ShaderStage::Fragment`, `gpu::BindGroupLayoutBuilder`)
- Forward-declare WebGPU types in headers; include `<webgpu/webgpu.h>` only in `.cpp`
- Builder pattern uses `&&`-qualified methods returning `std::move(*this)`
