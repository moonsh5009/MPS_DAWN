# core_render

> Rendering pipeline — camera, passes, post-processing, extension renderer interface.

## Module Structure

```
src/core_render/
├── CMakeLists.txt                          # STATIC library → mps::core_render (depends: core_util, core_gpu, core_platform)
├── render_types.h                          # Forward-declared WebGPU handles, render-specific enums
├── render_engine.h / render_engine.cpp     # Frame lifecycle orchestrator (BeginFrame/EndFrame)
├── object_renderer.h                       # IObjectRenderer interface (for extensions)
├── pipeline/
│   └── render_pipeline_builder.h/cpp       # Fluent render pipeline builder
├── camera/
│   ├── camera.h / camera.cpp               # Orbit camera (view + projection matrices)
│   └── camera_controller.h / .cpp          # Mouse-driven orbit/pan/zoom
├── uniform/
│   ├── camera_uniform.h / camera_uniform.cpp # Camera UBO sync to GPU
│   └── light_uniform.h / light_uniform.cpp   # Directional light UBO
├── pass/
│   ├── render_pass_builder.h / .cpp        # Render pass descriptor + execution
│   └── render_encoder.h / .cpp             # Unified RenderPassEncoder/RenderBundleEncoder
├── target/
│   └── render_target.h / render_target.cpp # Resizable depth/color texture target
├── geometry/
│   └── draw_command.h / draw_command.cpp   # Mesh binding + draw batch
└── post/
    ├── fullscreen_quad.h / fullscreen_quad.cpp # Screen-space triangle utility
    ├── fxaa_pass.h / fxaa_pass.cpp             # FXAA post-processing
    └── wboit_pass.h / wboit_pass.cpp           # Weighted blended OIT
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `RenderEngineConfig` | `render_engine.h` | Config: `clear_color`, `depth_format`, `enable_fxaa`, `enable_wboit` |
| `RenderEngine` | `render_engine.h` | Frame lifecycle: BeginFrame/EndFrame, sub-system orchestration |
| `IObjectRenderer` | `object_renderer.h` | Extension interface: Initialize, Resize, Render, Shutdown, GetOrder() |
| `RenderPipelineBuilder` | `pipeline/render_pipeline_builder.h` | Fluent render pipeline builder |
| `Camera` | `camera/camera.h` | Orbit camera with view/projection matrices |
| `CameraController` | `camera/camera_controller.h` | Mouse input → orbit/pan/zoom |
| `CameraUniform` | `uniform/camera_uniform.h` | Camera UBO (matrices, position, viewport) |
| `LightUniform` | `uniform/light_uniform.h` | Directional light UBO |
| `RenderPassBuilder` | `pass/render_pass_builder.h` | Render pass descriptor builder + execution |
| `RenderEncoder` | `pass/render_encoder.h` | Unified pass/bundle encoder wrapper |
| `RenderTarget` | `target/render_target.h` | Resizable GPU texture target |
| `DrawCommand` / `DrawList` | `geometry/draw_command.h` | Mesh binding + batched draw |
| `FullscreenQuad` | `post/fullscreen_quad.h` | Screen-space triangle for post-processing |
| `FXAAPass` | `post/fxaa_pass.h` | FXAA anti-aliasing post-process |
| `WBOITPass` | `post/wboit_pass.h` | Weighted blended OIT transparency |

## API

### RenderEngine

```cpp
void Initialize(WGPUSurface surface, uint32 width, uint32 height,
                const RenderEngineConfig& config = {});
void Shutdown();
void Resize(uint32 width, uint32 height);
void UpdateUniforms(float32 dt);

// Per-frame cycle
bool BeginFrame();
WGPUCommandEncoder GetEncoder() const;
WGPUTextureView GetFrameView() const;
void EndFrame();

// Sub-systems
gpu::SurfaceManager& GetSurface();
Camera& GetCamera();
CameraController& GetCameraController();
CameraUniform& GetCameraUniform();
LightUniform& GetLightUniform();
RenderTarget& GetDepthTarget();

// Post-processing
FXAAPass& GetFXAAPass();
WBOITPass& GetWBOITPass();

// Convenience
gpu::TextureFormat GetColorFormat() const;
gpu::TextureFormat GetDepthFormat() const;
uint32 GetWidth() const;
uint32 GetHeight() const;
```

### IObjectRenderer

```cpp
virtual const std::string& GetName() const = 0;
virtual void Initialize(RenderEngine& engine) {}
virtual void Resize(uint32 width, uint32 height) {}
virtual void Render(RenderEngine& engine, WGPURenderPassEncoder pass) = 0;
virtual void Shutdown() {}
virtual int32 GetOrder() const { return 1000; }
```

### RenderPassBuilder (fluent API)

```cpp
RenderPassBuilder(const std::string& label)
    .AddColorAttachment(view, load_op, store_op, clear_color)
    .SetDepthStencilAttachment(view, load_op, store_op, clear_depth)
    .Execute(encoder, [](WGPURenderPassEncoder pass) { ... });
```

## render_types.h Enums

Render-specific enums (values match WebGPU):

- `CullMode`, `FrontFace` — rasterizer state
- `LoadOp`, `StoreOp` — attachment operations
- `BlendFactor`, `BlendOp` — blend state
- `BlendState`, `ClearColor` — aggregate structs

GPU-general enums (`ShaderStage`, `BindingType`, `VertexFormat`, etc.) live in `core_gpu/gpu_types.h`.

## Shaders

| Directory | Shaders | Purpose |
|-----------|---------|---------|
| `assets/shaders/basic/` | `triangle.wgsl`, `mesh_vert.wgsl`, `mesh_frag.wgsl` | Basic geometry rendering |
| `assets/shaders/header/` | `camera.wgsl`, `light.wgsl` | Shared uniform struct definitions (imported by other shaders) |
| `assets/shaders/world/` | `default_vert.wgsl`, `default_frag.wgsl`, `wboit_frag.wgsl`, `wboit_compose.wgsl` | World-space rendering + WBOIT transparency |
| `assets/shaders/post/` | `fullscreen_vert.wgsl`, `fxaa_frag.wgsl` | Post-processing (fullscreen triangle + FXAA) |
